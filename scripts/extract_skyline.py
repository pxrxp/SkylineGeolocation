#!/usr/bin/env python
"""Extract skyline from a photo and overlay it on the original image."""
import sys, os
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# Import the DP skyline extractor and its helpers from archive
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archive.scripts.annotate_gsv import (
    compute_enhanced_skyline_feature_map,
    extract_pure_image_baseline_skyline,
)

def overlay_skyline(img_rgb, skyline_line, line_color=(0, 255, 255), line_thickness=3):
    """Draw the skyline boundary on the image as a colored line with sky tint."""
    overlay = img_rgb.copy()
    H, W = overlay.shape[:2]

    # Draw the skyline as a polyline
    points = np.column_stack((np.arange(W), skyline_line)).astype(np.int32)
    cv2.polylines(overlay, [points], isClosed=False, color=line_color, thickness=line_thickness)

    # Fill sky region with a semi-transparent blue tint
    mask = np.zeros((H, W), dtype=np.uint8)
    for c in range(W):
        r = int(skyline_line[c])
        if r > 0:
            mask[:r, c] = 255
    blue_overlay = overlay.copy()
    blue_overlay[mask == 255] = (40, 120, 255)
    overlay = cv2.addWeighted(overlay, 0.7, blue_overlay, 0.3, 0)

    # Re-draw the skyline line on top so it's crisp
    cv2.polylines(overlay, [points], isClosed=False, color=line_color, thickness=line_thickness)

    return overlay

if __name__ == "__main__":
    input_path = os.path.expanduser("~/FinalProjectDemo/tmp/captures/crop_0_20260829_170309.png")
    output_path = os.path.expanduser("~/FinalProjectDemo/tmp/captures/extracted.png")

    if not os.path.exists(input_path):
        print(f"Error: input image not found at {input_path}")
        sys.exit(1)

    print(f"Loading image: {input_path}")
    img_rgb = np.array(Image.open(input_path).convert("RGB"))
    H, W = img_rgb.shape[:2]
    print(f"  Image size: {W}x{H}")

    print("Extracting skyline via DP-based feature map...")
    skyline_line = extract_pure_image_baseline_skyline(img_rgb)
    print(f"  Raw skyline: {len(skyline_line)} columns, row range [{skyline_line.min()}, {skyline_line.max()}]")

    # Apply heavier Gaussian smoothing for a natural-looking profile
    # sigma=3.0 smooths ~15px features, keeps major terrain shapes
    skyline_line = gaussian_filter1d(skyline_line.astype(np.float64), sigma=3.0)
    # Second pass with moderate sigma for ultra-smooth result
    skyline_line = gaussian_filter1d(skyline_line, sigma=2.0)
    skyline_line = np.clip(np.round(skyline_line), 0, H - 1).astype(np.int32)
    print(f"  Smoothed skyline: row range [{skyline_line.min()}, {skyline_line.max()}]")

    print("Overlaying skyline on original image...")
    result = overlay_skyline(img_rgb, skyline_line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(result).save(output_path)
    print(f"Saved: {output_path}")
