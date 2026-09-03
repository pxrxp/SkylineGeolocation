#!/usr/bin/env python
"""Criteria Grid Browser — filter GSV panos by crop count and profile quality.

Top: curated panos (you pick from sidebar)
Middle: rejected panos grouped by reason
Bottom: all hand-annotated panos with per-image category picker

Usage:
    streamlit run scripts/criteria_dashboard.py
"""

import sys, glob
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import json
from PIL import Image

from src.query_profile import mask_to_boundary

MASKS_DIR = ROOT / "data" / "street_view" / "masks"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"
EVAL_FILE = ROOT / "data" / "street_view" / "gsv_improve_eval_results.json"

REJECT_CATEGORIES = [
    "NOT_ENOUGH_CROPS",
    "NO_SKYLINE",   # extract_elevation_profile status
    "LOW_CONFIDENCE",  # boundary coverage / reliable columns
    "FLAT",          # is_profile_applicable: std too low
    "LOW_RELIEF",    # is_profile_applicable: max elevation too low
    "GOOD",
]

st.set_page_config(page_title="Pipeline Filter Grid", layout="wide")


def _count_crops():
    """Count crop files per pano_id."""
    pano_crops = {}
    for f in CROPS_DIR.glob("*.png"):
        pid = f.name.rsplit("_h", 1)[0]
        pano_crops.setdefault(pid, []).append(f.name)
    return pano_crops


def _compute_profile_from_mask(mask, fov_y_deg=65.0, r_tilt=None, bin_deg=0.5):
    H, W = mask.shape
    binary = (mask < 128).astype(np.uint8)
    sky_ratio = float(binary.sum() / (H * W))
    if sky_ratio == 0.0 or sky_ratio == 1.0:
        return None, sky_ratio

    skyline_px = np.full(W, H - 1, dtype=np.float32)
    for c in range(W):
        sky_rows = np.where(binary[:, c] == 1)[0]
        if len(sky_rows) > 0:
            skyline_px[c] = sky_rows[0]

    import scipy.ndimage as ndimage
    skyline_px = ndimage.median_filter(skyline_px, size=5)

    aspect_ratio = W / H
    hfov_deg = np.degrees(2.0 * np.arctan(np.tan(np.radians(fov_y_deg) / 2.0) * aspect_ratio))
    focal_x = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    focal_y = H / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    x_c, y_c = W / 2.0, H / 2.0
    cols = np.arange(W)
    rays = np.vstack([(cols - x_c) / focal_x, (y_c - skyline_px) / focal_y, -np.ones(W)])
    rays /= np.linalg.norm(rays, axis=0)

    azim_cam = np.degrees(np.arctan2(rays[0, :], -rays[2, :]))
    if r_tilt is not None:
        rays = np.asarray(r_tilt) @ rays

    elev_deg = np.degrees(np.arcsin(np.clip(rays[1, :], -1.0, 1.0)))
    order = np.argsort(azim_cam)
    azim_cam, elev_deg = azim_cam[order], elev_deg[order]

    start_az = np.ceil(azim_cam[0] / bin_deg) * bin_deg
    end_az = np.floor(azim_cam[-1] / bin_deg) * bin_deg
    grid = np.arange(start_az, end_az + 1e-6, bin_deg)
    profile = np.interp(grid, azim_cam, elev_deg)
    return profile, sky_ratio


@st.cache_data
def load_all():
    gt_data = json.loads(GT_FILE.read_text()) if GT_FILE.exists() else {}
    eval_raw = json.loads(EVAL_FILE.read_text()) if EVAL_FILE.exists() else {}
    eval_results = {r["pano_id"]: r for r in eval_raw.get("results", [])}
    crop_counts = _count_crops()

    records = []
    for sid in gt_data:
        mask_path = MASKS_DIR / f"{sid}.png"
        if not mask_path.exists():
            continue

        n_crops = len(crop_counts.get(sid, []))
        if n_crops == 0:
            continue
        crop_files = sorted(glob.glob(str(CROPS_DIR / f"{sid}_*.png")))
        crop_path = crop_files[0] if crop_files else None

        mask = np.array(Image.open(mask_path).convert("L"))
        H, W = mask.shape

        sky_ratio = float(np.mean(mask < 128))
        boundary = mask_to_boundary(mask)
        boundary_coverage = float(np.mean(boundary >= 0))

        gt = gt_data.get(sid, {})
        cam_R = gt.get("cam_R_tilt")
        fov = gt.get("fov_y_deg", 65.0)
        profile, _ = _compute_profile_from_mask(mask, fov_y_deg=fov, r_tilt=cam_R)

        std_deg = float(np.std(profile)) if profile is not None and profile.size > 0 else 0.0
        max_elev = float(np.max(profile)) if profile is not None and profile.size > 0 else 0.0

        eval_r = eval_results.get(sid, {})

        records.append({
            "sid": sid,
            "crop_path": crop_path,
            "n_crops": n_crops,
            "sky_ratio": sky_ratio,
            "boundary_coverage": boundary_coverage,
            "std_deg": std_deg,
            "max_elev": max_elev,
            "err_baseline": eval_r.get("err_baseline"),
            "err_rrf": eval_r.get("err_rrf"),
            "coverage_deg": eval_r.get("coverage_deg", 0),
        })

    return pd.DataFrame(records)


def _render_cell(row, deleted_sids, thresholds, group_tag="", show_category=False):
    sid = row["sid"]
    t_bnd, t_std, t_elev = thresholds

    crop_path = row.get("crop_path")
    img_src = crop_path if (crop_path and Path(crop_path).exists()) else None
    if img_src:
        st.image(img_src, width="stretch")

    # SID + crop count + error
    n_crops = int(row.get("n_crops", 0))
    err_parts = [f"{n_crops} crop{'s' if n_crops != 1 else ''}"]
    if pd.notna(row.get("err_baseline")) and np.isfinite(row["err_baseline"]):
        err_parts.append(f"{row['err_baseline']:.0f}m")
    if pd.notna(row.get("err_rrf")) and np.isfinite(row["err_rrf"]):
        err_parts.append(f"rrf={row['err_rrf']:.0f}m")
    st.markdown(
        f"<div style='font-size:0.7em;font-weight:bold;margin-bottom:2px'>"
        f"{sid[:20]}… · {' · '.join(err_parts)}</div>",
        unsafe_allow_html=True,
    )



    if show_category:
        cat_key = f"cat_{group_tag}_{sid}"
        st.selectbox("Reason", REJECT_CATEGORIES, index=0, key=cat_key, label_visibility="collapsed")

    del_key = f"del_{group_tag}_{sid}"
    st.markdown(
        """<style>
        div[data-testid='stVerticalBlock'] button { font-size: 0; padding: 0; height: 0; overflow: hidden; }
        div[data-testid='stVerticalBlock']:hover button { font-size: 0.6em; padding: 0 0.3em; height: auto; }
        </style>""",
        unsafe_allow_html=True,
    )
    if st.button("✕", key=del_key, help="Hide"):
        deleted_sids.add(sid)
        st.rerun()


def show_grid(rows, cols_per_row, label="", show_top_n=5, deleted_sids=None,
              group_tag="", thresholds=None, show_category=False):
    if deleted_sids is None:
        deleted_sids = set()
    rows = rows[~rows["sid"].isin(deleted_sids)]
    if len(rows) == 0:
        return
    st.markdown(f"**{label}**")
    top = rows.head(show_top_n)
    rest = rows.iloc[show_top_n:]

    for start in range(0, len(top), cols_per_row):
        chunk = top.iloc[start : start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, (_, row) in enumerate(chunk.iterrows()):
            with cols[i]:
                _render_cell(row, deleted_sids, thresholds, group_tag=group_tag,
                             show_category=show_category)

    if len(rest) > 0:
        with st.expander("Show more…", expanded=False):
            for start in range(0, len(rest), cols_per_row):
                chunk = rest.iloc[start : start + cols_per_row]
                cols = st.columns(cols_per_row)
                for i, (_, row) in enumerate(chunk.iterrows()):
                    with cols[i]:
                        _render_cell(row, deleted_sids, thresholds, group_tag=group_tag,
                                     show_category=show_category)
    st.divider()


def main():
    st.title("Pipeline Filter Grid")

    if "deleted_sids" not in st.session_state:
        st.session_state.deleted_sids = set()
    if "curated_sids" not in st.session_state:
        st.session_state.curated_sids = []
    deleted_sids = st.session_state.deleted_sids

    with st.spinner("Loading panoramas..."):
        df = load_all()

    all_sids = df["sid"].tolist()

    # ── Sidebar ──
    st.sidebar.header("Filter Thresholds")
    min_crops = st.sidebar.slider("min crop count", 1, 5, 2, 1)
    min_boundary = st.sidebar.slider("min_boundary_coverage", 0.0, 1.0, 0.60, 0.05)
    min_std = st.sidebar.slider("min_std_deg", 0.0, 5.0, 1.5, 0.1)
    min_max_elev = st.sidebar.slider("min_max_elev_deg", 0.0, 10.0, 1.0, 0.1)
    thr = (min_boundary, min_std, min_max_elev)

    st.sidebar.header("View")
    cols_per_row = st.sidebar.slider("Grid columns", 3, 8, 5)
    max_per_group = st.sidebar.slider("Max per group", 5, 100, 30)

    st.sidebar.header("Curate good examples")
    curated = st.sidebar.multiselect("Pick panos", options=all_sids,
                                      default=st.session_state.curated_sids,
                                      key="curated_picker")
    st.session_state.curated_sids = curated

    if deleted_sids:
        if st.sidebar.button("Restore hidden"):
            st.session_state.deleted_sids.clear()
            st.rerun()

    # ── Curated section ──
    if curated:
        curated_df = df[df["sid"].isin(curated)].copy()
        if len(curated_df) > 0:
            show_grid(curated_df, cols_per_row,
                      label="⭐ Curated", deleted_sids=deleted_sids,
                      group_tag="curated", thresholds=thr)

    # ── Classify rejections (deduplicated) ──
    # Priority: NOT_ENOUGH_CROPS > NO_SKY/ALL_SKY > BOUNDARY > FLAT > LOW_RELIEF
    assigned = set()
    groups = []

    # Gate 1: not enough crops
    g = df[(df["n_crops"] < min_crops) & (~df.index.isin(assigned))]
    if len(g) > 0:
        g = g.sort_values("n_crops", ascending=True)
        groups.append((f"🚫 NOT_ENOUGH_CROPS ({min_crops}+ needed)", g.index.tolist()))
        assigned.update(g.index)

    # Gate 2: NO_SKYLINE (extract_elevation_profile)
    g = df[(~df.index.isin(assigned)) & ((df["sky_ratio"] == 0.0) | (df["sky_ratio"] == 1.0))]
    if len(g) > 0:
        groups.append(("🚫 NO_SKYLINE", g.index.tolist()))
        assigned.update(g.index)

    # Gate 3: LOW_CONFIDENCE — boundary coverage (extract_elevation_profile)
    g = df[(~df.index.isin(assigned)) & (df["boundary_coverage"] < min_boundary)]
    if len(g) > 0:
        g = g.sort_values("boundary_coverage", ascending=True)
        groups.append((f"🚫 LOW_CONFIDENCE (boundary < {min_boundary:.0%})", g.index.tolist()))
        assigned.update(g.index)

    # Gate 4: FLAT — is_profile_applicable: std too low
    g = df[(~df.index.isin(assigned)) & (df["std_deg"] < min_std)]
    if len(g) > 0:
        g = g.sort_values("std_deg", ascending=True)
        groups.append((f"🚫 FLAT (std < {min_std}°)", g.index.tolist()))
        assigned.update(g.index)

    # Gate 5: LOW_RELIEF — is_profile_applicable: max elevation too low
    g = df[(~df.index.isin(assigned)) & (df["max_elev"] < min_max_elev)]
    if len(g) > 0:
        g = g.sort_values("max_elev", ascending=True)
        groups.append((f"🚫 LOW_RELIEF (max < {min_max_elev}°)", g.index.tolist()))
        assigned.update(g.index)

    if groups:
        st.header("❌ Rejected")
        for i, (label, idx_list) in enumerate(groups):
            group_df = df.loc[idx_list].head(max_per_group)
            show_grid(group_df, cols_per_row,
                      label=label, deleted_sids=deleted_sids,
                      group_tag=f"g{i}", thresholds=thr)

    # ── All remaining panos at bottom ──
    remaining = df[~df.index.isin(assigned) & (~df["sid"].isin(set(curated or [])))]
    if len(remaining) > 0:
        st.header("📋 All remaining panos")
        st.caption("Hover to hide · Choose reject category per image")
        show_grid(remaining, cols_per_row,
                  label="Uncategorized", deleted_sids=deleted_sids,
                  group_tag="all", thresholds=thr, show_category=True)

    # ── Detail table ──
    with st.expander("Full data table", expanded=False):
        st.dataframe(
            df[["sid", "n_crops", "sky_ratio", "boundary_coverage", "std_deg", "max_elev"]],
            column_config={
                "n_crops": st.column_config.NumberColumn("Crops"),
                "sky_ratio": st.column_config.NumberColumn("Sky", format="%.0f%%"),
                "boundary_coverage": st.column_config.NumberColumn("Bnd", format="%.0f%%"),
                "std_deg": st.column_config.NumberColumn("Std°", format="%.2f"),
                "max_elev": st.column_config.NumberColumn("Elev°", format="%.2f"),
            },
            width="stretch",
        )


if __name__ == "__main__":
    main()
