#!/usr/bin/env python3
"""
VISUALIZATION SCRIPT - Visual Geo-Localization Alignment Check
=========================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from unified_evaluation_pipeline import extract_elevation_profile

# Load metadata and database
with open("data/synthetic_dataset_gt.json") as f: 
    gt_data = json.load(f)
db_local = np.load("data/horizon_db_local.npy")

def vertical_to_horizontal_fov(vertical_fov_deg, aspect_ratio=1.5):
    return np.degrees(2.0 * np.arctan(np.tan(np.radians(vertical_fov_deg) / 2.0) * aspect_ratio))

sample_id = 0
gt_info = gt_data[str(sample_id)]
true_idx = gt_info["closest_viewpoint_id"]
fov_y_deg = gt_info.get("fov_y_deg", 65.0)

image_path = f"data/synthetic_dataset/images/sample_{sample_id:04d}.png"
mask_path = f"data/synthetic_dataset/masks/sample_{sample_id:04d}.png"

# 1. Extract the query profile (ours) with correct vertical FOV and tilt compensation
r_tilt = gt_info.get("cam_R_tilt", None)
if r_tilt is not None:
    r_tilt = np.array(r_tilt, dtype=np.float32)

query_profile, start_az = extract_elevation_profile(mask_path, fov_y_deg=fov_y_deg, aspect_ratio=1.5, r_tilt=r_tilt)

# 2. Get the database profile (real DEM)
db_profile = db_local[true_idx]

# Plot the comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Panel 1: The Rendered Image
if os.path.exists(image_path):
    axes[0].imshow(Image.open(image_path))
    axes[0].set_title(f"1. Rendered Query Image (Sample {sample_id})")
    axes[0].axis('off')

# Panel 2: The Perfect Ground-Truth Mask
if os.path.exists(mask_path):
    axes[1].imshow(Image.open(mask_path), cmap='gray')
    axes[1].set_title("2. Perfect Ground-Truth Mask (Terrain=0, Sky=255)")
    axes[1].axis('off')

# Panel 3: Skyline Alignment Check
# We slide the query to its best matching position in the database to overlay them
m = len(query_profile)
db_padded = np.hstack([db_profile, db_profile[:m-1]])
q_norm = (query_profile - np.mean(query_profile)) / np.std(query_profile)

best_r = -1.0
best_off = 0
for off in range(1440):
    sub = db_padded[off : off + m]
    sub_norm = (sub - np.mean(sub)) / np.std(sub)
    r = np.mean(q_norm * sub_norm)
    if r > best_r:
        best_r = r
        best_off = off

matched_db_subsequence = db_padded[best_off : best_off + m]
matched_db_norm = (matched_db_subsequence - np.mean(matched_db_subsequence)) / np.std(matched_db_subsequence)

x_axis = np.arange(m) * 0.25
axes[2].plot(x_axis, q_norm, label="Extracted Skyline (Query)", color="crimson", lw=2)
axes[2].plot(x_axis, matched_db_norm, label="Database Skyline (DEM Grid Point)", color="royalblue", lw=2, linestyle="--")
axes[2].set_title(f"3. Profile Overlay (Pearson Correlation: {best_r:.4f})")
axes[2].set_xlabel("Relative Field of View (Degrees)")
axes[2].set_ylabel("Normalized Elevation")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
