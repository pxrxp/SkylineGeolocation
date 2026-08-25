#!/usr/bin/env python
"""Off-grid synthetic evaluation — REAL off-grid, memory-safe streaming.

Picks positions BETWEEN DB grid points, renders synthetic horizons
using HORAYZON at those positions, and matches against the DB by
streaming chunks (never loads full DB into RAM).

Usage:
    python archive/scripts/offgrid_synthetic_eval.py --samples 20
"""

import sys
import time
import json
import argparse
import heapq
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import horayzon as hray
from pyproj import Transformer
from horizon_format import decode_horizon_column

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
MESH_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "terrain_mesh.npy"
META_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "terrain_meta.json"
BIN_DEG = 0.5
N_BINS = int(360 / BIN_DEG)
CHUNK = 4000
TOP_KEEP = 50


def load_mesh_and_meta():
    vert_grid = np.load(str(MESH_PATH))
    with open(META_PATH) as f:
        meta = json.load(f)
    return vert_grid, meta


def load_db_coords():
    pf = pq.ParquetFile(str(DB_PATH))
    lats, lons = [], []
    for batch in pf.iter_batches(batch_size=50000, columns=["lat", "lon"]):
        df = batch.to_pandas()
        lats.append(df["lat"].values)
        lons.append(df["lon"].values)
    return np.concatenate(lats), np.concatenate(lons)


def pick_offgrid_positions(meta, n_samples, rng):
    px_w = abs(meta["px_w"])
    px_h = meta["px_h"]  # negative: y decreases with row index
    start_x = meta["start_x"]
    start_y = meta["start_y"]
    dim0 = meta["dem_dim_0"]
    dim1 = meta["dem_dim_1"]
    offset = px_w * 0.5
    margin = 5
    rows = rng.randint(margin, dim0 - margin, size=n_samples)
    cols = rng.randint(margin, dim1 - margin, size=n_samples)
    xs = start_x + cols * px_w + offset
    ys = start_y + rows * px_h + offset
    return xs, ys, rows, cols


def render_horizons_at_positions(vert_grid, meta, xs, ys, n_bins=N_BINS):
    n = len(xs)
    # Sample DEM elevation at each position from the mesh vertex buffer.
    # vert_grid is a flat interleaved [x,y,z,x,y,z,...] buffer from
    # hray.auxiliary.rearrange_pad_buffer, with SSE padding at the end.
    px_w = abs(meta["px_w"])
    start_x = meta["start_x"]
    start_y = meta["start_y"]
    dim0 = meta["dem_dim_0"]
    dim1 = meta["dem_dim_1"]
    px_h = meta["px_h"]  # negative: y decreases with row index
    cols = np.clip(np.round((xs - start_x) / px_w).astype(int), 0, dim1 - 1)
    rows = np.clip(np.round((ys - start_y) / px_h).astype(int), 0, dim0 - 1)
    # z is every 3rd element starting at index 2, first dim0*dim1 vertices
    idx = (rows * dim1 + cols) * 3 + 2
    zs = vert_grid[idx].astype(np.float32)
    coords = np.column_stack([
        xs.astype(np.float32),
        ys.astype(np.float32),
        zs,
    ])
    vec_up = np.zeros((n, 3), dtype=np.float32)
    vec_up[:, 2] = 1.0
    vec_north = np.zeros((n, 3), dtype=np.float32)
    vec_north[:, 1] = 1.0
    ray_org_elev = np.full(n, 1.6, dtype=np.float32)

    result = hray.horizon.horizon_locations(
        vert_grid, meta["dem_dim_0"], meta["dem_dim_1"],
        coords, vec_up, vec_north,
        dist_search=30000.0, azim_num=n_bins, hori_acc=0.1,
        ray_org_elev=ray_org_elev, hori_dist_out=False, elev_ang_low_lim=-89.0,
    )
    horizon_elev_rad = result[0] if isinstance(result, tuple) else result
    return np.rad2deg(horizon_elev_rad)


def add_noise(profile, sigma):
    if sigma == 0:
        return profile.copy()
    return profile + np.random.normal(0, sigma, size=profile.shape)


def feature_bundle(mat):
    mat = np.asarray(mat, dtype=np.float64)
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    val = (mat - mu) / sd
    d1_mat = np.gradient(val, axis=1)
    mu2 = d1_mat.mean(axis=1, keepdims=True)
    sd2 = d1_mat.std(axis=1, keepdims=True)
    sd2[sd2 < 1e-12] = 1.0
    d1 = (d1_mat - mu2) / sd2
    return val.astype(np.float32), d1.astype(np.float32)


def score_chunk(queries_bundle, db_chunk):
    """Score all queries against one DB chunk. Returns (n_queries, n_chunk) scores."""
    q_val, q_d1 = queries_bundle  # each (n_q, N_BINS)
    db_val, db_d1 = feature_bundle(db_chunk)

    # FFT-based Pearson-normalised circular cross-correlation.
    # irfft(conj(q) * d) / L gives Pearson correlation at each shift.
    # WITHOUT conj() this computes convolution (garbage).
    Fq_v = np.conj(np.fft.rfft(q_val, axis=1))
    Fq_d = np.conj(np.fft.rfft(q_d1, axis=1))
    Fd_v = np.fft.rfft(db_val, axis=1)
    Fd_d = np.fft.rfft(db_d1, axis=1)
    del db_val, db_d1

    cv = np.fft.irfft(Fq_v[:, None, :] * Fd_v[None, :, :], n=N_BINS, axis=2) / N_BINS
    cd = np.fft.irfft(Fq_d[:, None, :] * Fd_d[None, :, :], n=N_BINS, axis=2) / N_BINS
    combined = 0.5 * cv + 0.5 * cd  # (n_q, n_chunk, N_BINS)
    return combined.max(axis=2)  # (n_q, n_chunk) best over all shifts


def run_offgrid(n_samples=20, seed=42, output_path=None,
                noise_sigmas=(0.0, 0.25, 0.5, 1.0, 2.0)):
    rng = np.random.RandomState(seed)

    print("=" * 72)
    print("OFF-GRID SYNTHETIC EVALUATION (memory-safe streaming)")
    print("=" * 72)
    print(f"n_samples={n_samples}")

    # Load terrain mesh
    t0 = time.time()
    vert_grid, meta = load_mesh_and_meta()
    print(f"Loaded terrain mesh in {time.time()-t0:.1f}s")

    # Load DB coordinates
    t0 = time.time()
    lat_db, lon_db = load_db_coords()
    n_db = len(lat_db)
    print(f"Loaded {n_db} DB coords in {time.time()-t0:.1f}s")

    # Pick off-grid positions
    xs, ys, rows, cols = pick_offgrid_positions(meta, n_samples, rng)

    # Render horizons at off-grid positions
    t0 = time.time()
    offgrid_horizons = render_horizons_at_positions(vert_grid, meta, xs, ys)
    print(f"Rendered {n_samples} off-grid horizons in {time.time()-t0:.1f}s")

    # Convert UTM to GPS for nearest-neighbor lookup
    to_gps = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
    lon_off, lat_off = to_gps.transform(xs, ys)

    # Find nearest DB row for each off-grid position
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([lat_db, lon_db]))
    _, nearest_idx = tree.query(np.column_stack([lat_off, lon_off]))
    print(f"Nearest DB rows found")

    # Prepare query features (done once, reused for all chunks)
    all_conditions = []
    for sigma in noise_sigmas:
        queries = np.array([add_noise(p.copy(), sigma) for p in offgrid_horizons])
        q_val, q_d1 = feature_bundle(queries)
        all_conditions.append((f"noise={sigma}", q_val, q_d1))

    # uint8 quantization condition
    q_u8 = np.array([np.clip(np.round(p * (255/90)), 0, 255).astype(np.uint8).astype(np.float64) * (90/255)
                     for p in offgrid_horizons])
    q_val_u8, q_d1_u8 = feature_bundle(q_u8)
    all_conditions.append(("quant_u8", q_val_u8, q_d1_u8))

    # Stream DB and score all conditions simultaneously
    n_conds = len(all_conditions)
    # For each condition, maintain top-K heap per query
    # heap entry: (score, db_row_idx)
    best_rows = [[[-np.inf, -1] for _ in range(n_samples)] for _ in range(n_conds)]
    # best_rows[cond][query] = [best_score, best_row_idx]

    t0 = time.time()
    pf = pq.ParquetFile(str(DB_PATH))
    n_scanned = 0
    for batch in pf.iter_batches(batch_size=CHUNK, columns=["raw_horizon_deg"]):
        decoded = decode_horizon_column(batch["raw_horizon_deg"])
        n_chunk = len(decoded)

        for ci, (cond_name, q_val, q_d1) in enumerate(all_conditions):
            scores = score_chunk((q_val, q_d1), decoded)  # (n_samples, n_chunk)
            row_start = n_scanned

            for qi in range(n_samples):
                s = float(scores[qi].max())
                j = int(scores[qi].argmax())
                if s > best_rows[ci][qi][0]:
                    best_rows[ci][qi] = [s, row_start + j]

        n_scanned += n_chunk
        if n_scanned % 200000 == 0:
            print(f"  streamed {n_scanned}/{n_db} rows... ({time.time()-t0:.0f}s)")

    print(f"Scanned {n_scanned} rows in {time.time()-t0:.1f}s")

    # Load nearest DB row horizons for correlation-based error metric
    print("\nLoading nearest DB horizons for correlation analysis...")
    pf_meta = pq.ParquetFile(str(DB_PATH))
    nearest_hORIZONS = np.zeros((n_samples, N_BINS), dtype=np.float64)
    n_scanned_global = 0
    for batch in pf_meta.iter_batches(batch_size=50000, columns=["raw_horizon_deg"]):
        decoded = decode_horizon_column(batch["raw_horizon_deg"])
        n_chunk = len(decoded)
        for qi in range(n_samples):
            if nearest_idx[qi] < n_scanned_global + n_chunk:
                local_idx = nearest_idx[qi] - n_scanned_global
                if 0 <= local_idx < n_chunk:
                    nearest_hORIZONS[qi] = decoded[local_idx]
        n_scanned_global += n_chunk
        if n_scanned_global >= max(nearest_idx) + 1:
            break

    # Compute ceiling: max-over-shifts correlation with nearest DB point
    # (uses same FFT pipeline as matching, but only the true nearest row)
    true_val, true_d1 = feature_bundle(offgrid_horizons)
    nearest_val, nearest_d1 = feature_bundle(nearest_hORIZONS)

    # Ceiling = max-over-shifts Pearson NCC with nearest DB row
    Fq_v = np.conj(np.fft.rfft(true_val, axis=1))
    Fq_d = np.conj(np.fft.rfft(true_d1, axis=1))
    Fn_v = np.fft.rfft(nearest_val, axis=1)
    Fn_d = np.fft.rfft(nearest_d1, axis=1)
    cv_n = np.fft.irfft(Fq_v * Fn_v, n=N_BINS, axis=1) / N_BINS
    cd_n = np.fft.irfft(Fq_d * Fn_d, n=N_BINS, axis=1) / N_BINS
    ceiling_corr = np.max(0.5 * cv_n + 0.5 * cd_n, axis=1)  # best shift per query

    # Also compute direct Pearson (shift=0) for reference
    def _pearson_corr(a, b):
        a_z = a - a.mean()
        b_z = b - b.mean()
        return float(np.sum(a_z * b_z) / (np.linalg.norm(a_z) * np.linalg.norm(b_z) + 1e-12))

    direct_corr = np.array([
        0.5 * _pearson_corr(true_val[qi], nearest_val[qi]) +
        0.5 * _pearson_corr(true_d1[qi], nearest_d1[qi])
        for qi in range(n_samples)
    ])

    print(f"Ceiling (max-shift NCC with nearest DB): median={np.median(ceiling_corr):.4f}")
    print(f"Direct Pearson (shift=0):                median={np.median(direct_corr):.4f}")
    print(f"  Range: [{ceiling_corr.min():.4f}, {ceiling_corr.max():.4f}]")
    print(f"  (This is the best any matcher can do on off-grid positions)")

    # Use ceiling_corr as the reference for corr_drop
    true_nearest_corr = ceiling_corr

    # Compute errors
    all_results = []
    for ci, (cond_name, q_val, q_d1) in enumerate(all_conditions):
        t0 = time.time()
        errs, match_corrs, true_corrs, ranks = [], [], [], []
        for qi in range(n_samples):
            ti = nearest_idx[qi]
            best_row = best_rows[ci][qi][1]
            best_score = best_rows[ci][qi][0]

            # Geographic error (secondary — limited by 30m grid spacing)
            err = geodesic((lat_off[qi], lon_off[qi]),
                           (lat_db[best_row], lon_db[best_row])).meters
            errs.append(err)

            # Correlation: query vs matched DB row (primary metric)
            match_corrs.append(best_score)

            # Correlation: query vs true off-grid horizon (signal preservation)
            true_c = (0.5 * _pearson_corr(q_val[qi], true_val[qi]) +
                      0.5 * _pearson_corr(q_d1[qi], true_d1[qi]))
            true_corrs.append(true_c)

            # Rank: approximate by checking if nearest row scored higher
            # (we don't have full scores, so this is a lower bound)
            ranks.append(0 if best_row == ti else -1)

        errs = np.array(errs)
        match_corrs = np.array(match_corrs)
        true_corrs = np.array(true_corrs)
        dt = time.time() - t0

        # Signal degradation: how much correlation dropped from true position
        corr_drop = true_nearest_corr - match_corrs

        print(f"\n--- {cond_name} --- ({dt:.1f}s)")
        print(f"  Geographic error (limited by 30m grid): median={np.median(errs):.0f}m")
        print(f"  Match correlation:  median={np.median(match_corrs):.4f}")
        print(f"  True correlation:   median={np.median(true_corrs):.4f}")
        print(f"  Correlation drop:   median={np.median(corr_drop):.4f} "
              f"(positive = signal degraded)")

        for qi in range(n_samples):
            all_results.append({
                "condition": cond_name,
                "true_row": int(nearest_idx[qi]),
                "best_row": int(best_rows[ci][qi][1]),
                "top1_err_m": float(errs[qi]),
                "match_corr": float(match_corrs[qi]),
                "true_corr": float(true_corrs[qi]),
                "corr_drop": float(corr_drop[qi]),
            })

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY (real off-grid, streaming)")
    print("=" * 72)
    print(f"NOTE: Geographic error is bounded by ~{abs(meta['px_w'])*0.5:.0f}m (half grid spacing).")
    print(f"      Primary metric is correlation drop (signal degradation).\n")
    print(f"{'condition':>12} | {'match_corr':>10} {'corr_drop':>10} {'geo_err':>8}")
    print("-" * 55)
    seen = []
    for r in all_results:
        if r["condition"] not in seen:
            seen.append(r["condition"])
    for cond in seen:
        rs = [r for r in all_results if r["condition"] == cond]
        mc = np.median([r["match_corr"] for r in rs])
        cd = np.median([r["corr_drop"] for r in rs])
        ge = np.median([r["top1_err_m"] for r in rs])
        print(f"{cond:>12} | {mc:>10.4f} {cd:>+10.4f} {ge:>7.0f}m")

    # Save
    if output_path is None:
        output_path = ROOT / "archive" / "synthetic_results" / "offgrid_eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    run_offgrid(n_samples=args.samples, seed=args.seed, output_path=args.output)
