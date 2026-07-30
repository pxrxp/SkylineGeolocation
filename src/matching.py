"""Core matching utilizing fastdtw, scipy.stats, and scipy.signal against a single database."""
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import correlate
from scipy.stats import zscore
from fastdtw import fastdtw


def _safe_zscore(x):
    x = np.asarray(x, dtype=np.float64)
    if x.std() < 1e-12:
        return np.zeros_like(x)
    return zscore(x)


def _feature_bundle(profile):
    """Extract and z-score normalize features using standard scipy.stats utilities."""
    profile = np.asarray(profile, dtype=np.float64)
    value = _safe_zscore(profile)
    d1 = _safe_zscore(np.gradient(value))
    d2 = _safe_zscore(np.gradient(d1))
    return value, d1, d2


def fft_prefilter(db_matrix, query_profile, bin_deg,
                  weights=(0.33, 0.33, 0.33),
                  expected_offset_deg=None, tolerance_deg=None):
    """
    Scores database horizons against query_profile using cross-correlation.
    """
    n_horizons, horizon_length = db_matrix.shape
    profile_length = len(query_profile)

    query_val, query_d1, query_d2 = _feature_bundle(query_profile)

    offset_mask = None
    if expected_offset_deg is not None:
        bins = np.arange(horizon_length)
        expected_bin = (expected_offset_deg / bin_deg) % horizon_length
        tolerance_bins = tolerance_deg / bin_deg
        circular_dist = np.minimum(
            (bins - expected_bin) % horizon_length,
            (expected_bin - bins) % horizon_length
        )
        offset_mask = circular_dist <= tolerance_bins

    best_corr = np.zeros(n_horizons)
    best_offset = np.zeros(n_horizons, dtype=np.int32)

    for i in range(n_horizons):
        horizon = db_matrix[i].astype(np.float64)
        h_val, h_d1, h_d2 = _feature_bundle(horizon)

        pad_val = np.hstack([h_val, h_val[:profile_length - 1]])
        pad_d1 = np.hstack([h_d1, h_d1[:profile_length - 1]])
        pad_d2 = np.hstack([h_d2, h_d2[:profile_length - 1]])
        corr_val = correlate(pad_val, query_val, mode="valid")
        corr_d1 = correlate(pad_d1, query_d1, mode="valid")
        corr_d2 = correlate(pad_d2, query_d2, mode="valid")
        corr = weights[0] * corr_val + weights[1] * corr_d1 + weights[2] * corr_d2
        if offset_mask is not None:
            corr = np.where(offset_mask[:len(corr)], corr, -np.inf)
        best_bin = int(np.argmax(corr))
        best_offset[i] = best_bin
        best_corr[i] = corr[best_bin]

    return best_corr, best_offset


def _compute_confidence(corr_scores, dtw_costs, top_k=5):
    """Compute confidence metrics from raw scores."""
    n = len(corr_scores)
    if n == 0:
        return {"best_score": 0.0, "second_score": 0.0, "score_gap": 0.0, "ambiguous": True}

    # Normalize: higher is better for combined score
    # combined = fft_corr - dtw_cost_normalized
    # Simple heuristic: rank by combined metric
    scores = np.array(corr_scores)
    best = float(scores[0]) if len(scores) > 0 else 0.0
    second = float(scores[1]) if len(scores) > 1 else 0.0

    return {
        "best_score": best,
        "second_score": second,
        "score_gap": float(best - second),
        "ambiguous": (best - second) < 0.03,
    }


def match_query(db_matrix, bin_deg, query_profile, top_k=30, dtw_window=15,
                valid_vp_mask=None, expected_offset_deg=None,
                tolerance_deg=20.0, weights=(0.33, 0.33, 0.33),
                spatial_stride=5,
                min_corr=0.15, min_score_gap=0.03):
    """
    Two-stage matching directly on a single database matrix.

    Returns:
        dict with keys: ok, status, reason, matches, confidence, diagnostics
    """
    if db_matrix is None or db_matrix.size == 0:
        return {
            "ok": False, "status": "INVALID_INPUT",
            "reason": "Empty database matrix",
            "matches": [], "confidence": {"ambiguous": True},
            "diagnostics": {},
        }

    profile_length = len(query_profile)
    n_viewpoints = db_matrix.shape[0]

    if profile_length < 10:
        return {
            "ok": False, "status": "INVALID_QUERY",
            "reason": f"Query profile too short ({profile_length} bins)",
            "matches": [], "confidence": {"ambiguous": True},
            "diagnostics": {"profile_length": profile_length},
        }

    if np.any(~np.isfinite(query_profile)):
        return {
            "ok": False, "status": "INVALID_QUERY",
            "reason": "Query contains NaN or Inf values",
            "matches": [], "confidence": {"ambiguous": True},
            "diagnostics": {},
        }

    # 1. Coarse Spatial Search
    coarse_indices = np.arange(0, n_viewpoints, spatial_stride)
    coarse_db = db_matrix[coarse_indices]
    coarse_vp_mask = None
    if valid_vp_mask is not None:
        coarse_vp_mask = valid_vp_mask[coarse_indices]

    corr_scores, offsets = fft_prefilter(
        coarse_db, query_profile, bin_deg,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
        weights=weights
    )
    if coarse_vp_mask is not None:
        corr_scores = np.where(coarse_vp_mask, corr_scores, -np.inf)

    # 2. Local Fine Refinement
    top_coarse_indices = np.argsort(-corr_scores)[:5]
    fine_candidate_set = set()
    for idx in top_coarse_indices:
        if corr_scores[idx] == -np.inf:
            continue
        global_coarse_idx = coarse_indices[idx]
        local_range = np.arange(
            max(0, global_coarse_idx - spatial_stride),
            min(n_viewpoints, global_coarse_idx + spatial_stride + 1)
        )
        fine_candidate_set.update(local_range)
    fine_indices = np.array(sorted(fine_candidate_set), dtype=np.int32)

    fine_db = db_matrix[fine_indices]
    fine_vp_mask = None
    if valid_vp_mask is not None:
        fine_vp_mask = valid_vp_mask[fine_indices]
    fine_corr, fine_offsets = fft_prefilter(
        fine_db, query_profile, bin_deg,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
        weights=weights
    )
    if fine_vp_mask is not None:
        fine_corr = np.where(fine_vp_mask, fine_corr, -np.inf)

    query_val, query_d1, _ = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T

    # 3. DTW Alignment on top-k fine candidates
    candidates = []
    top_k_fine = np.argsort(-fine_corr)[:top_k]
    for idx in top_k_fine:
        if fine_corr[idx] == -np.inf:
            continue
        global_vp_idx = fine_indices[idx]
        horizon = db_matrix[global_vp_idx]
        offset = fine_offsets[idx]
        windowed = horizon[np.arange(offset, offset + profile_length) % len(horizon)]
        db_val, db_d1, _ = _feature_bundle(windowed)
        db_features = np.vstack([db_val, db_d1]).T
        dtw_cost, _ = fastdtw(query_features, db_features, radius=dtw_window, dist=euclidean)
        candidates.append({
            "viewpoint_idx": int(global_vp_idx),
            "fft_corr": float(fine_corr[idx]),
            "dtw_cost": float(dtw_cost),
            "offset_bin": int(offset),
        })

    if not candidates:
        return {
            "ok": False, "status": "NO_MATCH",
            "reason": "No candidates survived filtering",
            "matches": [], "confidence": {"ambiguous": True},
            "diagnostics": {"n_coarse": len(coarse_indices), "n_fine": len(fine_indices)},
        }

    candidates.sort(key=lambda c: c["dtw_cost"] / max(0.01, c["fft_corr"]))

    # Build matches list
    matches = []
    for c in candidates:
        combined = c["fft_corr"] - 0.01 * c["dtw_cost"]
        matches.append({
            "row_index": c["viewpoint_idx"],
            "score": float(combined),
            "fft_corr": c["fft_corr"],
            "dtw_distance": c["dtw_cost"],
            "offset_deg": float(c["offset_bin"] * bin_deg),
        })

    confidence = _compute_confidence(
        [m["fft_corr"] for m in matches[:5]],
        [m["dtw_distance"] for m in matches[:5]],
    )

    status = "OK"
    reason = "Match found"
    ok = True
    if confidence["ambiguous"] and confidence["best_score"] < min_corr:
        status = "LOW_CONFIDENCE"
        reason = f"Low correlation ({confidence['best_score']:.3f} < {min_corr}) and ambiguous top-2 gap"
        ok = False
    elif confidence["ambiguous"]:
        status = "LOW_CONFIDENCE"
        reason = f"Ambiguous top-2 score gap ({confidence['score_gap']:.4f} < {min_score_gap})"
        ok = False

    diagnostics = {
        "n_coarse": int(len(coarse_indices)),
        "n_fine": int(len(fine_indices)),
        "n_candidates": int(len(candidates)),
        "profile_length": int(profile_length),
        "bin_deg": float(bin_deg),
    }

    return {
        "ok": ok,
        "status": status,
        "reason": reason,
        "matches": matches,
        "confidence": confidence,
        "diagnostics": diagnostics,
    }


def finalize_matches(result_view, query_profile, dtw_window):
    """Refinement helper using fastdtw on streamed candidate row-views."""
    query_val, query_d1, _ = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T

    candidates = []
    for corr, _, vp_idx, offset, horizon_window in result_view:
        db_val, db_d1, _ = _feature_bundle(horizon_window)
        db_features = np.vstack([db_val, db_d1]).T
        dtw_cost, _ = fastdtw(query_features, db_features, radius=dtw_window, dist=euclidean)
        candidates.append({
            "viewpoint_idx": int(vp_idx),
            "fft_corr": float(corr),
            "dtw_cost": float(dtw_cost),
            "offset_bin": int(offset),
        })

    candidates.sort(key=lambda c: c["dtw_cost"] / max(0.01, c["fft_corr"]))
    return candidates
