"""Multi-Photo Perspective Fusion Evaluator.

Fuses multiple perspective crops from the same panorama (e.g. Photo 1 @ Heading 0° + Photo 2 @ Heading 90°)
into a wide-FOV joint query profile to break valley symmetry and evaluate matching against the DEM database.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matching import match_query
from query_profile import extract_elevation_profile

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"


def load_crop_pairs():
    """Group crops by panorama ID into multi-perspective pairs."""
    if not CROPS_DIR.exists():
        return {}

    meta_files = sorted(CROPS_DIR.glob("*.json"))
    panos = {}

    for mf in meta_files:
        if mf.name == "crop_state.json" or mf.name == "forbidden_regions.json":
            continue
        with open(mf) as f:
            meta = json.load(f)
        pid = meta["pano_id"]
        if pid not in panos:
            panos[pid] = []
        img_path = CROPS_DIR / meta["filename"]
        if img_path.exists():
            meta["img_path"] = str(img_path)
            panos[pid].append(meta)

    # Keep panoramas with at least 2 crops
    return {k: v for k, v in panos.items() if len(v) >= 2}


def fuse_multi_perspective_profiles(crop_list):
    """Combine profiles from multiple crops into a single wide-FOV joint profile array."""
    profiles = []

    for c in crop_list:
        img = np.array(Image.open(c["img_path"]).convert("RGB"))
        res = extract_elevation_profile(
            img,
            fov_y_deg=c.get("fov_y_deg", 65.0),
            bin_deg=0.5,
        )
        if res["ok"]:
            prof = res["profile"]
            profiles.append((c["heading_deg"], prof))

    if not profiles:
        return None

    # Sort profiles by heading angle
    profiles.sort(key=lambda x: x[0])

    # Construct continuous 360-degree array initialized to NaN
    n_bins = 720  # 360 degrees @ 0.5 deg resolution
    joint_horizon = np.full(n_bins, np.nan, dtype=np.float32)

    for heading, prof in profiles:
        m = len(prof)
        center_bin = int(round((heading % 360.0) / 0.5))
        half_m = m // 2
        for i in range(m):
            bin_idx = (center_bin - half_m + i) % n_bins
            val = prof[i]
            if not np.isnan(val):
                joint_horizon[bin_idx] = val

    return joint_horizon


def main():
    print("Multi-Photo Perspective Fusion Evaluator")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"Error: DB file not found: {DB_PATH}")
        return

    pairs = load_crop_pairs()
    print(f"Found {len(pairs)} panoramas with multi-photo crops in {CROPS_DIR}")

    if not pairs:
        print("No multi-crop panoramas found. Use gsv_crop_dashboard.py to save multiple perspective crops per pano.")
        return

    pq_file = pq.ParquetFile(DB_PATH)

    for pid, crop_list in list(pairs.items())[:5]:
        print(f"\nProcessing Pano ID: {pid}")
        headings = [f"{c['heading_deg']:.0f}°" for c in crop_list]
        print(f"  Available crop headings: {', '.join(headings)}")

        joint_profile = fuse_multi_perspective_profiles(crop_list)
        if joint_profile is None:
            print("  Failed to extract joint horizon profile.")
            continue

        valid_bins = int((~np.isnan(joint_profile)).sum())
        coverage_deg = valid_bins * 0.5
        print(f"  Fused Joint FOV Coverage: {coverage_deg:.1f}° ({valid_bins}/720 bins)")

        # Run matching
        matches = match_query(
            joint_profile,
            pq_file,
            top_k=5,
        )

        if matches["ok"]:
            print("  Top-3 Matches:")
            for rank, m in enumerate(matches["matches"][:3], start=1):
                print(
                    f"    Rank {rank}: VP {m['vp_idx']} | "
                    f"Score: {m['score']:.4f} | "
                    f"Lat: {m['lat']:.5f}, Lon: {m['lon']:.5f}"
                )
        else:
            print(f"  Matching failed: {matches['reason']}")


if __name__ == "__main__":
    from PIL import Image

    main()