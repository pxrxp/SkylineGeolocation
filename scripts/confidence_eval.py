#!/usr/bin/env python
"""
Confidence-Gated Multi-Photo Evaluator V4
==========================================
Uses PRECOMPUTED DB features + official ncc_scores for fast (~2s) scanning
of ALL 1.34M VPs with stride=1.

Confidence = profile distinctiveness:
  How many DB locations look similar to this query?
  Few = distinctive = trustworthy match
  Many = generic = reject

Requires: python scripts/precompute_db_features.py
"""
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matching import ncc_scores
from query_profile import extract_elevation_profile

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
VAL_PATH = ROOT / "data" / "street_view" / "db_val.npy"
D1_PATH = ROOT / "data" / "street_view" / "db_d1.npy"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
ANNOT_FILE = ROOT / "data" / "street_view" / "annotations.json"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"

W, H = 1080, 720
BIN_DEG = 0.5
N_BINS = int(360.0 / BIN_DEG)
CHUNK = 8000
TOP_N = 500

# --- Load precomputed features ---
_db_val = None
_db_d1 = None
_n_rows = 0


def _load_features():
    global _db_val, _db_d1, _n_rows
    if _db_val is not None:
        return
    import os
    print("Loading precomputed DB features...")
    val_bytes = os.path.getsize(str(VAL_PATH))
    _n_rows = val_bytes // (N_BINS * 4)
    _db_val = np.memmap(str(VAL_PATH), dtype=np.float32, mode='r',
                         shape=(_n_rows, N_BINS))
    _db_d1 = np.memmap(str(D1_PATH), dtype=np.float32, mode='r',
                         shape=(_n_rows, N_BINS))
    print(f"  {_n_rows} rows x {N_BINS} bins x 2 features "
          f"({_n_rows * N_BINS * 8 / 1e9:.2f} GB)")


def fast_scan(query_profile):
    """Scan ALL DB rows using precomputed features. Returns correlation array."""
    _load_features()
    n_corr = np.zeros(_n_rows, dtype=np.float64)
    for start in range(0, _n_rows, CHUNK):
        end = min(start + CHUNK, _n_rows)
        db_v = np.array(_db_val[start:end], dtype=np.float64)
        db_d = np.array(_db_d1[start:end], dtype=np.float64)
        best_corr, _ = ncc_scores(db_v, db_d, query_profile, bin_deg=BIN_DEG)
        n_corr[start:end] = best_corr
    return n_corr


def fast_scan_top(query_profile, top_n=TOP_N):
    """Scan ALL DB rows, return top-N with lat/lon."""
    _load_features()
    heap = []
    pf = pq.ParquetFile(str(DB_PATH))

    # We need to scan all rows and keep top-N
    row_offset = 0
    for batch in pf.iter_batches(batch_size=CHUNK, columns=["lat", "lon"]):
        end = min(row_offset + CHUNK, _n_rows)
        n = end - row_offset

        db_v = np.array(_db_val[row_offset:end], dtype=np.float64)
        db_d = np.array(_db_d1[row_offset:end], dtype=np.float64)
        best_corr, _ = ncc_scores(db_v, db_d, query_profile, bin_deg=BIN_DEG)

        lats = batch.to_pandas()["lat"].values[:n]
        lons = batch.to_pandas()["lon"].values[:n]

        for i in range(n):
            score = float(best_corr[i])
            entry = (score, row_offset + i, float(lats[i]), float(lons[i]))
            if len(heap) < top_n:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

        row_offset = end

    return sorted(heap, key=lambda x: -x[0])


def mask_from_points(points):
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.unique(xs).size < 2:
        return None
    cols = np.arange(W, dtype=np.float64)
    rows_interp = np.interp(cols, xs, ys)
    rows_interp = np.clip(np.rint(rows_interp), 1, H - 1).astype(int)
    rr = np.arange(H)[:, None]
    sky = rr < rows_interp[None, :]
    return np.where(sky, 0, 255).astype(np.uint8)


def extract_crop_profiles(crops, gt_data):
    profiles = []
    for c in crops:
        mask = mask_from_points(c["points"])
        if mask is None:
            continue
        fov_y = c.get("fov_y_deg", 65.0)
        sid = c.get("sid", "")
        gt_entry = gt_data.get(sid) or {}
        r_tilt = c.get("cam_R_tilt") or gt_entry.get("cam_R_tilt")
        if r_tilt is not None:
            r_tilt = np.array(r_tilt)
        skyline_rows = np.full(W, H - 1, dtype=np.int32)
        for col in range(W):
            sky_rows = np.where(mask[:, col] == 0)[0]
            if len(sky_rows) > 0:
                skyline_rows[col] = sky_rows[-1]
        unclipped_cols = (skyline_rows > 2) & (skyline_rows < H - 2)
        res = extract_elevation_profile(
            mask, fov_y_deg=fov_y, r_tilt=r_tilt, bin_deg=BIN_DEG,
            column_keep_mask=unclipped_cols, azim_frame="camera",
        )
        if not res["ok"]:
            continue
        profiles.append((c.get("heading_deg", 0.0), res["profile"]))
    return profiles


def fuse_profiles(crop_profiles):
    joint = np.full(N_BINS, np.nan, dtype=np.float64)
    for heading, prof in crop_profiles:
        m = len(prof)
        center_bin = int(round((heading % 360.0) / BIN_DEG))
        half_m = m // 2
        for i in range(m):
            bin_idx = (center_bin - half_m + i) % N_BINS
            if not np.isnan(prof[i]):
                joint[bin_idx] = prof[i]
    valid = ~np.isnan(joint)
    if valid.sum() < 30:
        return None, 0
    all_bins = np.arange(N_BINS)
    fused = np.interp(all_bins, all_bins[valid], joint[valid])
    return fused, int(valid.sum())


def evaluate_pano(gt_data, pano_id, crops):
    t0 = time.time()

    crop_profiles = extract_crop_profiles(crops, gt_data)
    if len(crop_profiles) < 2:
        return None

    fused, coverage = fuse_profiles(crop_profiles)
    if fused is None:
        return None

    fov = coverage * BIN_DEG

    # Full stride=1 scan with precomputed features (~2-3s)
    t_scan = time.time()
    all_corrs = fast_scan(fused)
    dt_scan = time.time() - t_scan

    # Distinctiveness metrics
    n70 = int(np.sum(all_corrs > 0.70))
    n75 = int(np.sum(all_corrs > 0.75))
    n80 = int(np.sum(all_corrs > 0.80))
    n_total = len(all_corrs)

    sorted_corrs = np.sort(all_corrs)[::-1]
    top1 = float(sorted_corrs[0]) if n_total > 0 else 0
    mean_top10 = float(np.mean(sorted_corrs[:10])) if n_total >= 10 else top1
    mean_top50 = float(np.mean(sorted_corrs[:50])) if n_total >= 50 else top1
    gap_top10 = top1 - mean_top10

    # Get top candidates with lat/lon
    t_top = time.time()
    top_cands = fast_scan_top(fused, top_n=TOP_N)
    dt_top = time.time() - t_top

    if not top_cands:
        return None

    top_score, top_idx, top_lat, top_lon = top_cands[0]

    # Ground truth
    gt = gt_data.get(pano_id) or {}
    true_lat = gt.get("true_lat") or gt.get("lat")
    true_lon = gt.get("true_lon") or gt.get("lon")

    err = float("inf")
    if true_lat is not None:
        err = geodesic((true_lat, true_lon), (top_lat, top_lon)).meters

    # True VP rank in top candidates
    true_rank = -1
    if true_lat is not None:
        for sc, _, r_lat, r_lon in top_cands:
            if abs(r_lat - true_lat) < 0.0001 and abs(r_lon - true_lon) < 0.0001:
                true_rank = sum(1 for s, _, _, _ in top_cands if s > sc + 0.001)
                break
        if true_rank == -1:
            true_rank = TOP_N

    # --- Confidence gates ---
    reject_reasons = []

    # Gate 1: FOV wide enough to break valley symmetry
    if fov < 200.0:
        reject_reasons.append(f"fov={fov:.0f}<200")

    # Gate 2: Profile must be DISTINCTIVE
    # n70 = how many of 1.34M VPs have corr > 0.70
    # Distinctive: < 200 matches (profile is unique in the DB)
    # Generic: > 200 matches (profile looks like thousands of locations)
    if n70 > 200:
        reject_reasons.append(f"generic(n70={n70})")

    # Gate 3: Top candidate must clearly beat the field
    if gap_top10 < 0.02:
        reject_reasons.append(f"low_gap10({gap_top10:.4f})")

    is_confident = len(reject_reasons) == 0
    tag = "[CONFIDENT]" if is_confident else "[REJECT  ]"

    dt = time.time() - t0
    hit = " HIT" if err < 1000 else ""
    print(f"  {tag} {pano_id[:18]:18s} crops={len(crop_profiles):1d} "
          f"FOV={fov:5.0f}° err={err:8.0f}m "
          f"corr={top_score:.3f} n70={n70:5d} gap10={gap_top10:.4f} "
          f"rank={true_rank:6d}{hit} "
          f"[scan={dt_scan:.1f}s top={dt_top:.1f}s total={dt:.0f}s]")
    if reject_reasons:
        print(f"           REASON: {'; '.join(reject_reasons)}")

    return {
        "sid": pano_id, "fov": fov, "err": err,
        "fused_corr": top_score, "gap_top10": gap_top10,
        "n70": n70, "n75": n75, "n80": n80,
        "n_crops": len(crop_profiles),
        "is_confident": is_confident,
        "reject_reasons": reject_reasons,
        "true_rank": true_rank,
        "dt_scan": dt_scan, "dt_top": dt_top,
    }


def main():
    print("=" * 72)
    print("CONFIDENCE-GATED MULTI-PHOTO EVALUATOR V4")
    print("=" * 72)
    print("\nPrecomputed features → ~2-3s scan per pano, stride=1, ALL 1.34M VPs")
    print("Confidence = profile distinctiveness (few DB matches = trustworthy)\n")

    _load_features()

    with open(GT_FILE) as f:
        gt_data = json.load(f)
    with open(ANNOT_FILE) as f:
        annots = json.load(f).get("annotations", {})

    # Group crops by pano
    panos = {}
    for sid, points in annots.items():
        meta_p = CROPS_DIR / f"{sid}.json"
        if not meta_p.exists():
            continue
        with open(meta_p) as f:
            meta = json.load(f)
        pid = meta.get("pano_id")
        if not pid:
            continue
        if pid not in panos:
            panos[pid] = []
        meta["sid"] = sid
        meta["points"] = points
        panos[pid].append(meta)

    multi = {k: v for k, v in panos.items() if len(v) >= 2}
    print(f"Found {len(multi)} multi-crop panoramas\n")

    results = []
    for pano_id, crops in sorted(multi.items()):
        r = evaluate_pano(gt_data, pano_id, crops)
        if r is not None:
            results.append(r)

    # --- Summary ---
    confident = [r for r in results if r["is_confident"]]
    rejected = [r for r in results if not r["is_confident"]]
    all_errs = [r["err"] for r in results if r["err"] < float("inf")]

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    def print_stats(label, errs):
        if not errs:
            print(f"  {label}: N=0")
            return
        errs = np.array(errs)
        print(f"  {label}: N={len(errs)}")
        print(f"    Median error: {np.median(errs):.0f}m")
        print(f"    Mean error:   {np.mean(errs):.0f}m")
        for thr in [100, 500, 1000, 2000, 5000, 10000]:
            n = sum(1 for e in errs if e < thr)
            print(f"    < {thr:5d}m: {n}/{len(errs)} ({100*n/len(errs):5.1f}%)")

    print_stats("ALL PANOS", all_errs)
    print()
    conf_errs = [r["err"] for r in confident if r["err"] < float("inf")]
    print_stats("CONFIDENT (gate passed)", conf_errs)
    print()
    rej_errs = [r["err"] for r in rejected if r["err"] < float("inf")]
    print_stats("REJECTED (gate failed)", rej_errs)

    n_hits = sum(1 for e in all_errs if e < 1000)
    n_hits_conf = sum(1 for r in confident if r["err"] < 1000)
    n_hits_rej = sum(1 for r in rejected if r["err"] < 1000)

    print(f"\n  PRECISION: {n_hits_conf}/{max(len(confident),1)} confident panos true "
          f"({100*n_hits_conf/max(len(confident),1):.1f}%)")
    print(f"  RECALL: {n_hits_conf}/{max(n_hits,1)} hits captured "
          f"({100*n_hits_conf/max(n_hits,1):.0f}%)")
    print(f"  ACCEPTANCE: {len(confident)}/{len(results)} "
          f"({100*len(confident)/max(len(results),1):.1f}%)")
    if n_hits_rej > 0:
        print(f"  FALSE NEGATIVES (rejected hits): {n_hits_rej}")

    # --- Distinctiveness analysis ---
    print("\n" + "=" * 72)
    print("DISTINCTIVENESS ANALYSIS")
    print("=" * 72)
    hits = [r for r in results if r["err"] < 1000]
    misses = [r for r in results if r["err"] >= 1000 and r["err"] < float("inf")]

    for metric in ["n70", "n75", "n80", "gap_top10", "fused_corr"]:
        h_vals = [r[metric] for r in hits]
        m_vals = [r[metric] for r in misses]
        if h_vals and m_vals:
            print(f"\n  {metric}:")
            print(f"    HITS ({len(h_vals)}): mean={np.mean(h_vals):.4f} "
                  f"med={np.median(h_vals):.4f} "
                  f"[{np.min(h_vals):.4f}, {np.max(h_vals):.4f}]")
            print(f"    MISS ({len(m_vals)}): mean={np.mean(m_vals):.4f} "
                  f"med={np.median(m_vals):.4f} "
                  f"[{np.min(m_vals):.4f}, {np.max(m_vals):.4f}]")

    # --- Rejection reasons ---
    if rejected:
        print(f"\n  REJECTION REASONS:")
        reason_counts = {}
        for r in rejected:
            for reason in r["reject_reasons"]:
                key = reason.split("=")[0].split("(")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
        for key, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {key:25s}: {count}/{len(rejected)}")

    # --- Confident detail ---
    if confident:
        print("\n" + "=" * 72)
        print("CONFIDENT PANOS")
        print("=" * 72)
        for r in sorted(confident, key=lambda x: x["err"]):
            status = "HIT" if r["err"] < 1000 else "FALSE+"
            print(f"  {r['sid'][:22]:22s} err={r['err']:8.0f}m "
                  f"corr={r['fused_corr']:.3f} n70={r['n70']:5d} [{status}]")

    # Save
    out_path = ROOT / "data" / "street_view" / "confidence_eval_v4.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
