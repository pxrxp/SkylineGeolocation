"""Extracts and validates 1D elevation-angle skyline profiles from terrain masks."""

import numpy as np
import cv2
from PIL import Image
import scipy.ndimage as ndimage


def is_profile_applicable(profile, min_std_deg=1.5, min_max_elev_deg=1.0):
    """
    Evaluates whether a horizon profile contains sufficient topographic variation
    and vertical relief to be reliably matched.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if profile.size == 0:
        return False, "Empty profile"
    if not np.all(np.isfinite(profile)):
        return False, "Profile contains NaN or Inf values"

    std_val = np.std(profile)
    max_val = np.max(profile)

    if std_val < min_std_deg:
        return False, f"Profile too flat (std={std_val:.2f}° < {min_std_deg}°)"

    if max_val < min_max_elev_deg:
        return (
            False,
            f"Insufficient terrain relief above horizontal (max={max_val:.2f}° < {min_max_elev_deg}°)",
        )

    return True, "Valid topographic profile"


def extract_elevation_profile(
    mask_path,
    fov_y_deg=65.0,
    aspect_ratio=None,
    r_tilt=None,
    bin_deg=0.25,
    min_boundary_coverage=0.5,
    column_keep_mask=None,
    azim_frame="camera",
):
    """
    Translates a binary sky-terrain mask into a 1D elevation-angle profile
    projected onto a uniform azimuth grid.

    `mask_path` may be a file path (string) or a (H, W) numpy array / PIL Image
    (sky=0/black, terrain=255/white convention).

    Returns:
        dict with keys: ok, status, reason, profile, start_az, diagnostics
    """
    if isinstance(mask_path, str):
        try:
            pil_img = Image.open(mask_path).convert("L")
            mask = np.array(pil_img)
        except Exception as e:
            return {
                "ok": False,
                "status": "INVALID_INPUT",
                "reason": f"Cannot open mask: {e}",
                "profile": None,
                "start_az": None,
                "diagnostics": {},
            }
    elif isinstance(mask_path, np.ndarray):
        mask = np.asarray(mask_path, dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[2] > 1 else mask[:, :, 0]
    elif isinstance(mask_path, Image.Image):
        mask = np.array(mask_path.convert("L"))
    else:
        return {
            "ok": False,
            "status": "INVALID_INPUT",
            "reason": f"mask_path must be a path or array, got {type(mask_path).__name__}",
            "profile": None,
            "start_az": None,
            "diagnostics": {},
        }

    H, W = mask.shape
    if H < 2 or W < 2:
        return {
            "ok": False,
            "status": "INVALID_INPUT",
            "reason": f"Mask too small ({H}x{W})",
            "profile": None,
            "start_az": None,
            "diagnostics": {"width": W, "height": H},
        }

    aspect_ratio = aspect_ratio or (W / H)
    sky_is_white = np.mean(mask[:10, :]) > np.mean(mask[-10:, :])
    binary = (
        (mask < 128).astype(np.uint8)
        if sky_is_white
        else (mask >= 128).astype(np.uint8)
    )

    sky_ratio = float(binary.sum() / (H * W))
    if sky_ratio == 0.0:
        return {
            "ok": False,
            "status": "NO_SKYLINE",
            "reason": "No sky pixels found in mask",
            "profile": None,
            "start_az": None,
            "diagnostics": {
                "width": W,
                "height": H,
                "sky_ratio": 0.0,
                "sky_is_white": bool(sky_is_white),
            },
        }
    if sky_ratio == 1.0:
        return {
            "ok": False,
            "status": "NO_SKYLINE",
            "reason": "Mask is all sky, no terrain boundary",
            "profile": None,
            "start_az": None,
            "diagnostics": {
                "width": W,
                "height": H,
                "sky_ratio": 1.0,
                "sky_is_white": bool(sky_is_white),
            },
        }

    skyline_px = np.full(W, H - 1, dtype=np.float32)
    missing_cols = 0
    for c in range(W):
        sky_rows = np.where(binary[:, c] == 1)[0]
        if len(sky_rows) > 0:
            skyline_px[c] = sky_rows[0]
        else:
            missing_cols += 1
            skyline_px[c] = H - 1

    boundary_coverage = 1.0 - (missing_cols / W)
    skyline_px = ndimage.median_filter(skyline_px, size=5)

    kept_frac = 1.0
    if column_keep_mask is not None:
        keep = np.asarray(column_keep_mask, dtype=bool)
        if keep.ndim == 1 and len(keep) == W:
            kept_frac = float(keep.mean())
            if kept_frac < min_boundary_coverage:
                return {
                    "ok": False,
                    "status": "LOW_CONFIDENCE",
                    "reason": f"Too few reliable columns after filtering ({kept_frac:.3f} < {min_boundary_coverage})",
                    "profile": None,
                    "start_az": None,
                    "diagnostics": {
                        "width": W,
                        "height": H,
                        "kept_fraction": kept_frac,
                    },
                }
            skyline_px = np.where(keep, skyline_px, H - 1)

    hfov_deg = np.degrees(
        2.0 * np.arctan(np.tan(np.radians(fov_y_deg) / 2.0) * aspect_ratio)
    )
    focal_x = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    focal_y = H / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    x_c, y_c = W / 2.0, H / 2.0
    cols = np.arange(W)
    rays = np.vstack(
        [(cols - x_c) / focal_x, (y_c - skyline_px) / focal_y, -np.ones(W)]
    )
    rays /= np.linalg.norm(rays, axis=0)

    # Camera-frame azimuth from unrotated rays (forward = -z), before tilt.
    azim_cam = np.degrees(np.arctan2(rays[0, :], -rays[2, :]))

    if r_tilt is not None:
        rays = np.asarray(r_tilt) @ rays

    elev_deg = np.degrees(np.arcsin(np.clip(rays[1, :], -1.0, 1.0)))

    if azim_frame == "world":
        azim_deg = np.degrees(np.arctan2(rays[0, :], -rays[2, :]))
    else:
        azim_deg = azim_cam

    if column_keep_mask is not None and kept_frac > 0:
        sel = skyline_px < H - 1
        if sel.sum() < 3:
            return {
                "ok": False,
                "status": "LOW_CONFIDENCE",
                "reason": "Fewer than 3 reliable skyline columns after filtering",
                "profile": None,
                "start_az": None,
                "diagnostics": {
                    "width": W,
                    "height": H,
                    "kept_columns": int(sel.sum()),
                },
            }
        azim_deg = azim_deg[sel]
        elev_deg = elev_deg[sel]

    order = np.argsort(azim_deg)
    azim_deg, elev_deg = azim_deg[order], elev_deg[order]

    start_az = np.ceil(azim_deg[0] / bin_deg) * bin_deg
    end_az = np.floor(azim_deg[-1] / bin_deg) * bin_deg
    grid = np.arange(start_az, end_az + 1e-6, bin_deg)
    profile = np.interp(grid, azim_deg, elev_deg)

    diagnostics = {
        "width": W,
        "height": H,
        "sky_ratio": sky_ratio,
        "sky_is_white": bool(sky_is_white),
        "boundary_coverage": float(boundary_coverage),
        "missing_columns": int(missing_cols),
        "kept_fraction": float(kept_frac),
        "hfov_deg": float(hfov_deg),
        "fov_y_deg": float(fov_y_deg),
        "bin_deg": float(bin_deg),
        "profile_std_deg": float(np.std(profile)),
        "profile_max_deg": float(np.max(profile)),
        "profile_min_deg": float(np.min(profile)),
        "profile_length": int(len(profile)),
    }

    if boundary_coverage < min_boundary_coverage:
        return {
            "ok": False,
            "status": "LOW_CONFIDENCE",
            "reason": f"Boundary coverage too low ({boundary_coverage:.3f} < {min_boundary_coverage})",
            "profile": profile,
            "start_az": float(start_az),
            "diagnostics": diagnostics,
        }

    applicable, msg = is_profile_applicable(profile)
    if not applicable:
        return {
            "ok": False,
            "status": "LOW_CONFIDENCE",
            "reason": msg,
            "profile": profile,
            "start_az": float(start_az),
            "diagnostics": diagnostics,
        }

    return {
        "ok": True,
        "status": "OK",
        "reason": "Valid profile extracted",
        "profile": profile,
        "start_az": float(start_az),
        "diagnostics": diagnostics,
    }


def evaluate_skyline_quality(img, mask, profile, boundary_gradient_threshold=15.0):
    """Quality gate for sky-terrain masks before matching.

    Parameters
    ----------
    img : ndarray (H, W, 3) uint8 — the cropped photo (RGB).
    mask : ndarray (H, W) uint8 — binary mask (sky=0, terrain=255).
    profile : ndarray (L,) float — extracted elevation-angle profile.
    boundary_gradient_threshold : float — median gradient below this → reject.

    Returns
    -------
    (passed, score, reason) — bool, float quality score [0,1], str.
    """
    import cv2

    H, W = mask.shape
    if H < 2 or W < 2:
        return False, 0.0, "MASK_TOO_SMALL"

    # --- Terrain relief check (reuses is_profile_applicable logic) ---
    profile = np.asarray(profile, dtype=np.float64)
    if profile.size == 0:
        return False, 0.0, "EMPTY_PROFILE"
    if not np.all(np.isfinite(profile)):
        return False, 0.0, "PROFILE_HAS_NAN"
    if np.std(profile) < 1.5:
        return False, 0.0, "FLAT_TERRAIN_NO_RELIEF"

    # --- Boundary edge strength (fog / haze detection) ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()

    # Find the skg.copy()

    # Find the sky/terrain boundary pixel per column
    boundary_row = np.full(W, H, dtype=np.int32)
    for c in range(W):
        sky_rows = np.where(mask[:, c] == 0)[0]  # sky=0
        if len(sky_rows) > 0:
            boundary_row[c] = sky_rows[0]

    valid = boundary_row < H
    if valid.sum() < W * 0.3:
        return False, 0.0, "BOUNDARY_COVERAGE_TOO_LOW"

    # Boundary hugging the image top (steep-uphill shot): no sky above the
    # boundary to create contrast, so the fog gradient is unmeasurable.
    # Skip the fog check rather than reject a legitimate clear shot.
    median_boundary_row = float(np.median(boundary_row[valid]))
    if median_boundary_row >= 20.0:
        # Sobel gradient magnitude at boundary pixels (use ±5px window)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        boundary_vals_list = []
        for i, c in enumerate(np.where(valid)[0]):
            br = boundary_row[c]
            lo = max(0, br - 5)
            hi = min(H, br + 6)
            boundary_vals_list.append(float(np.max(grad_mag[lo:hi, c])))
        boundary_vals = np.array(boundary_vals_list)
        median_grad = float(np.median(boundary_vals))

        if median_grad < boundary_gradient_threshold:
            return False, 0.0, "FOG_OR_HAZE_OBSCURATION"
        fog_score = min(median_grad / 60.0, 1.0)
    else:
        fog_score = 1.0

    # Score: normalize median gradient to [0, 1], capped at 60
    return True, fog_score, "OK"


def compute_column_keep_mask(img, mask, gradient_threshold=8.0):
    """Per-column reliability mask: exclude fuzzy (tree/cloud) boundary columns.

    Columns whose boundary edge strength (max Sobel gradient within a ±5px
    window of the boundary) falls below the threshold are marked unreliable
    and excluded from profile extraction.

    Parameters
    ----------
    img : ndarray (H, W, 3) uint8 RGB.
    mask : ndarray (H, W) uint8 — sky=0, terrain=255.
    gradient_threshold : float — columns with boundary gradient below this
        are dropped (default 8.0; calibrated for DCP-dehazed inputs).

    Returns
    -------
    keep : ndarray (W,) bool — True for reliable columns.
    per_column_gradient : ndarray (W,) float64 — boundary gradient per column.
    """
    H, W = mask.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()

    boundary_row = np.full(W, H, dtype=np.int32)
    for c in range(W):
        sky_rows = np.where(mask[:, c] == 0)[0]
        if len(sky_rows) > 0:
            boundary_row[c] = sky_rows[0]

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    per_col = np.zeros(W, dtype=np.float64)
    for c in range(W):
        br = boundary_row[c]
        if br >= H:
            per_col[c] = 0.0
            continue
        lo = max(0, br - 5)
        hi = min(H, br + 6)
        per_col[c] = float(np.max(grad_mag[lo:hi, c]))

    keep = (boundary_row < H) & (per_col >= gradient_threshold)
    return keep, per_col
