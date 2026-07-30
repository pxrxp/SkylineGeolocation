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
        return False, f"Insufficient terrain relief above horizontal (max={max_val:.2f}° < {min_max_elev_deg}°)"

    return True, "Valid topographic profile"


def extract_elevation_profile(mask_path, fov_y_deg=65.0, aspect_ratio=None, r_tilt=None, bin_deg=0.25,
                              min_boundary_coverage=0.5):
    """
    Translates a binary sky-terrain mask into a 1D elevation-angle profile
    projected onto a uniform azimuth grid.

    Returns:
        dict with keys: ok, status, reason, profile, start_az, diagnostics
    """
    if not isinstance(mask_path, str) and not hasattr(mask_path, 'startswith'):
        return {
            "ok": False,
            "status": "INVALID_INPUT",
            "reason": "mask_path must be a string path",
            "profile": None, "start_az": None, "diagnostics": {}
        }

    try:
        pil_img = Image.open(mask_path).convert("L")
        mask = np.array(pil_img)
    except Exception as e:
        return {
            "ok": False,
            "status": "INVALID_INPUT",
            "reason": f"Cannot open mask: {e}",
            "profile": None, "start_az": None, "diagnostics": {}
        }

    H, W = mask.shape
    if H < 2 or W < 2:
        return {
            "ok": False,
            "status": "INVALID_INPUT",
            "reason": f"Mask too small ({H}x{W})",
            "profile": None, "start_az": None, "diagnostics": {"width": W, "height": H}
        }

    aspect_ratio = aspect_ratio or (W / H)
    sky_is_white = np.mean(mask[:10, :]) > np.mean(mask[-10:, :])
    binary = (mask < 128).astype(np.uint8) if sky_is_white else (mask >= 128).astype(np.uint8)

    sky_ratio = float(binary.sum() / (H * W))
    if sky_ratio == 0.0:
        return {
            "ok": False, "status": "NO_SKYLINE",
            "reason": "No sky pixels found in mask",
            "profile": None, "start_az": None,
            "diagnostics": {"width": W, "height": H, "sky_ratio": 0.0, "sky_is_white": bool(sky_is_white)}
        }
    if sky_ratio == 1.0:
        return {
            "ok": False, "status": "NO_SKYLINE",
            "reason": "Mask is all sky, no terrain boundary",
            "profile": None, "start_az": None,
            "diagnostics": {"width": W, "height": H, "sky_ratio": 1.0, "sky_is_white": bool(sky_is_white)}
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

    hfov_deg = np.degrees(2.0 * np.arctan(np.tan(np.radians(fov_y_deg) / 2.0) * aspect_ratio))
    focal_x = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    focal_y = H / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    x_c, y_c = W / 2.0, H / 2.0
    cols = np.arange(W)
    rays = np.vstack([(cols - x_c) / focal_x, (y_c - skyline_px) / focal_y, -np.ones(W)])
    rays /= np.linalg.norm(rays, axis=0)

    if r_tilt is not None:
        rays = np.asarray(r_tilt) @ rays

    elev_deg = np.degrees(np.arcsin(np.clip(rays[1, :], -1.0, 1.0)))
    azim_deg = np.degrees(np.arctan2(rays[0, :], -rays[2, :]))

    order = np.argsort(azim_deg)
    azim_deg, elev_deg = azim_deg[order], elev_deg[order]

    start_az = np.ceil(azim_deg[0] / bin_deg) * bin_deg
    end_az = np.floor(azim_deg[-1] / bin_deg) * bin_deg
    grid = np.arange(start_az, end_az + 1e-6, bin_deg)
    profile = np.interp(grid, azim_deg, elev_deg)

    diagnostics = {
        "width": W, "height": H,
        "sky_ratio": sky_ratio,
        "sky_is_white": bool(sky_is_white),
        "boundary_coverage": float(boundary_coverage),
        "missing_columns": int(missing_cols),
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
            "ok": False, "status": "LOW_CONFIDENCE",
            "reason": f"Boundary coverage too low ({boundary_coverage:.3f} < {min_boundary_coverage})",
            "profile": profile, "start_az": float(start_az),
            "diagnostics": diagnostics,
        }

    applicable, msg = is_profile_applicable(profile)
    if not applicable:
        return {
            "ok": False, "status": "LOW_CONFIDENCE",
            "reason": msg,
            "profile": profile, "start_az": float(start_az),
            "diagnostics": diagnostics,
        }

    return {
        "ok": True, "status": "OK",
        "reason": "Valid profile extracted",
        "profile": profile, "start_az": float(start_az),
        "diagnostics": diagnostics,
    }
