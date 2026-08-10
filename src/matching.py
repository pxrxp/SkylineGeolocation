"""Vectorized, normalized skyline matching against a horizon database.

Uses Pearson-normalized cross-correlation for the coarse spatial scan
(replacing the per-row Python loop with batched NumPy operations) and
fastdtw for fine refinement. All scores are properly normalised so that
thresholds are interpretable and defensible.
"""

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from fastdtw import fastdtw

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def _safe_zscore(x):
    """Z-score that returns zeros for constant arrays."""
    x = np.asarray(x, dtype=np.float64)
    if x.std() < 1e-12:
        return np.zeros_like(x)
    return zscore(x)


def _safe_zscore_matrix(mat):
    """Z-score each row of a 2-D array independently."""
    mat = np.asarray(mat, dtype=np.float64)
    means = mat.mean(axis=1, keepdims=True)
    stds = mat.std(axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    return (mat - means) / stds


def _feature_bundle(profile):
    """Value + first derivative, z-scored."""
    profile = np.asarray(profile, dtype=np.float64)
    value = _safe_zscore(profile)
    d1 = _safe_zscore(np.gradient(value))
    return value, d1


def _feature_bundle_matrix(mat):
    """Batch feature bundle: returns (value, d1) each (N, L)."""
    mat = np.asarray(mat, dtype=np.float64)
    value = _safe_zscore_matrix(mat)
    d1 = _safe_zscore_matrix(np.gradient(mat, axis=1))
    return value, d1


# ---------------------------------------------------------------------------
# Pearson-normalised circular cross-correlation (vectorised)
# ---------------------------------------------------------------------------


def _pearson_ncc_batch(db_ext, query_zm, q_norm):
    """Pearson NCC between every (N, L) db position and one query.

    Parameters
    ----------
    db_ext : (N, L+M-1) float64
        Circularly extended db feature array.
    query_zm : (M,) float64
        Zero-mean query feature.
    q_norm : float
        L2-norm of query_zm.

    Returns
    -------
    ncc : (N, L) float64
        Pearson correlation at every offset.
    """
    N, ext_len = db_ext.shape
    M = len(query_zm)
    L = ext_len - M + 1

    # Sliding windows: (N, L, M) view — no extra memory
    windows = np.lib.stride_tricks.sliding_window_view(db_ext, M, axis=1)

    # Cross-correlation (numerator): q · window for each offset
    numer = windows @ query_zm  # (N, L)

    # Denominator: L2-norm of each M-length window via cumulative sums
    cum = np.concatenate(
        [np.zeros((N, 1), dtype=np.float64), np.cumsum(db_ext, axis=1)], axis=1
    )
    cum_sq = np.concatenate(
        [np.zeros((N, 1), dtype=np.float64), np.cumsum(db_ext**2, axis=1)], axis=1
    )

    win_sum = cum[:, M : M + L] - cum[:, :L]
    win_sq_sum = cum_sq[:, M : M + L] - cum_sq[:, :L]
    win_var = win_sq_sum - win_sum**2 / M
    win_norm = np.sqrt(np.maximum(win_var, 0.0))

    denom = q_norm * win_norm
    ncc = numer / np.maximum(denom, 1e-12)
    return ncc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def feature_bundle_matrix(db_matrix):
    """Batch feature bundle: returns (value, d1) each (N, L). Public so the
    evaluation loop can compute DB features once per chunk and reuse them
    across all queries in a batch."""
    return _feature_bundle_matrix(np.asarray(db_matrix, dtype=np.float64))


def ncc_scores(
    db_val,
    db_d1,
    query_profile,
    bin_deg,
    weights=(0.5, 0.5),
    expected_offset_deg=None,
    tolerance_deg=None,
):
    """Pearson NCC of one query against precomputed DB features.

    db_val, db_d1 : (N, L) feature arrays from `feature_bundle_matrix`.
    """
    query_profile = np.asarray(query_profile, dtype=np.float64)
    N, L = db_val.shape
    M = len(query_profile)

    db_ext_val = np.concatenate([db_val, db_val[:, : M - 1]], axis=1)
    db_ext_d1 = np.concatenate([db_d1, db_d1[:, : M - 1]], axis=1)

    q_val, q_d1 = _feature_bundle(query_profile)
    q_val_zm = q_val - q_val.mean()
    q_d1_zm = q_d1 - q_d1.mean()

    ncc_val = _pearson_ncc_batch(db_ext_val, q_val_zm, np.linalg.norm(q_val_zm))
    ncc_d1 = _pearson_ncc_batch(db_ext_d1, q_d1_zm, np.linalg.norm(q_d1_zm))

    combined = weights[0] * ncc_val + weights[1] * ncc_d1

    # --- Compass offset mask ---
    if expected_offset_deg is not None and tolerance_deg is not None:
        bins = np.arange(L)
        expected_bin = (expected_offset_deg / bin_deg) % L
        tolerance_bins = tolerance_deg / bin_deg
        circular_dist = np.minimum((bins - expected_bin) % L, (expected_bin - bins) % L)
        mask = circular_dist <= tolerance_bins
        combined = np.where(mask[np.newaxis, :], combined, -np.inf)

    best_offset = np.argmax(combined, axis=1).astype(np.int32)
    best_corr = combined[np.arange(N), best_offset]
    return best_corr, best_offset


def fft_prefilter(
    db_matrix,
    query_profile,
    bin_deg,
    weights=(0.5, 0.5),
    expected_offset_deg=None,
    tolerance_deg=None,
):
    """Vectorised Pearson NCC prefilter over a database chunk.

    Parameters
    ----------
    db_matrix : (N, L) float array — raw elevation-angle horizons.
    query_profile : (M,) float array — query elevation-angle profile.
    bin_deg : float — angular bin size in degrees (typically 1.0).
    weights : tuple — (weight_value, weight_d1). Sum need not be 1.
    expected_offset_deg : float or None — compass-derived expected offset.
    tolerance_deg : float or None — compass tolerance in degrees.

    Returns
    -------
    best_corr : (N,) float64 — best Pearson NCC per row.
    best_offset : (N,) int32 — best offset bin per row.
    """
    db_val, db_d1 = feature_bundle_matrix(db_matrix)
    return ncc_scores(
        db_val,
        db_d1,
        query_profile,
        bin_deg,
        weights=weights,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
    )


def _compute_confidence(matches):
    """Compute calibrated confidence from a sorted matches list.

    Returns dict with best_score, second_score, score_gap, ambiguous.
    """
    if not matches:
        return {
            "best_score": 0.0,
            "second_score": 0.0,
            "score_gap": 0.0,
            "ambiguous": True,
        }

    best_score = matches[0]["score"]
    second_score = matches[1]["score"] if len(matches) > 1 else 0.0
    score_gap = best_score - second_score
    return {
        "best_score": best_score,
        "second_score": second_score,
        "score_gap": score_gap,
        "ambiguous": score_gap < 0.03,
    }


def match_query(
    db_matrix,
    bin_deg,
    query_profile,
    top_k=30,
    dtw_window=15,
    valid_vp_mask=None,
    expected_offset_deg=None,
    tolerance_deg=20.0,
    weights=(0.5, 0.5),
    spatial_stride=5,
    min_corr=0.3,
    min_score_gap=0.03,
):
    """Vectorised two-stage matching: coarse Pearson NCC + DTW refinement.

    Returns
    -------
    dict with ok, status, reason, matches, confidence, diagnostics.
    """
    if db_matrix is None or db_matrix.size == 0:
        return {
            "ok": False,
            "status": "INVALID_INPUT",
            "reason": "Empty database matrix",
            "matches": [],
            "confidence": {"ambiguous": True},
            "diagnostics": {},
        }

    profile_length = len(query_profile)
    n_viewpoints = db_matrix.shape[0]

    if profile_length < 10:
        return {
            "ok": False,
            "status": "INVALID_QUERY",
            "reason": f"Query profile too short ({profile_length} bins)",
            "matches": [],
            "confidence": {"ambiguous": True},
            "diagnostics": {"profile_length": profile_length},
        }

    if np.any(~np.isfinite(query_profile)):
        return {
            "ok": False,
            "status": "INVALID_QUERY",
            "reason": "Query contains NaN or Inf values",
            "matches": [],
            "confidence": {"ambiguous": True},
            "diagnostics": {},
        }

    # 1. Coarse spatial search (vectorised)
    coarse_indices = np.arange(0, n_viewpoints, spatial_stride)
    coarse_db = db_matrix[coarse_indices]
    coarse_vp_mask = (
        valid_vp_mask[coarse_indices] if valid_vp_mask is not None else None
    )

    corr_scores, offsets = fft_prefilter(
        coarse_db,
        query_profile,
        bin_deg,
        weights=weights,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
    )
    if coarse_vp_mask is not None:
        corr_scores = np.where(coarse_vp_mask, corr_scores, -np.inf)

    # 2. Local fine refinement around top-5 coarse hits
    top_coarse_indices = np.argsort(-corr_scores)[:5]
    fine_candidate_set = set()
    for idx in top_coarse_indices:
        if corr_scores[idx] == -np.inf:
            continue
        global_coarse_idx = coarse_indices[idx]
        local_range = np.arange(
            max(0, global_coarse_idx - spatial_stride),
            min(n_viewpoints, global_coarse_idx + spatial_stride + 1),
        )
        fine_candidate_set.update(local_range)
    fine_indices = np.array(sorted(fine_candidate_set), dtype=np.int32)
    fine_db = db_matrix[fine_indices]
    fine_vp_mask = valid_vp_mask[fine_indices] if valid_vp_mask is not None else None

    fine_corr, fine_offsets = fft_prefilter(
        fine_db,
        query_profile,
        bin_deg,
        weights=weights,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
    )
    if fine_vp_mask is not None:
        fine_corr = np.where(fine_vp_mask, fine_corr, -np.inf)

    # 3. DTW on top-k fine candidates
    query_val, query_d1 = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T

    candidates = []
    top_k_fine = np.argsort(-fine_corr)[:top_k]
    for idx in top_k_fine:
        if fine_corr[idx] == -np.inf:
            continue
        global_vp_idx = fine_indices[idx]
        horizon = db_matrix[global_vp_idx]
        offset = fine_offsets[idx]
        windowed = horizon[np.arange(offset, offset + profile_length) % len(horizon)]
        db_val, db_d1 = _feature_bundle(windowed)
        db_features = np.vstack([db_val, db_d1]).T
        dtw_cost, _ = fastdtw(
            query_features, db_features, radius=dtw_window, dist=euclidean
        )
        dtw_normalized = dtw_cost / max(len(query_features), 1)
        candidates.append(
            {
                "viewpoint_idx": int(global_vp_idx),
                "fft_corr": float(fine_corr[idx]),
                "dtw_cost": float(dtw_cost),
                "dtw_normalized": float(dtw_normalized),
                "offset_bin": int(offset),
            }
        )

    if not candidates:
        return {
            "ok": False,
            "status": "NO_MATCH",
            "reason": "No candidates survived filtering",
            "matches": [],
            "confidence": {"ambiguous": True},
            "diagnostics": {
                "n_coarse": int(len(coarse_indices)),
                "n_fine": int(len(fine_indices)),
            },
        }

    # Score: Pearson corr minus small DTW penalty (defensible blend)
    for c in candidates:
        c["score"] = c["fft_corr"] - 0.01 * c["dtw_normalized"]

    candidates.sort(key=lambda c: c["score"], reverse=True)

    matches = []
    for c in candidates:
        matches.append(
            {
                "row_index": c["viewpoint_idx"],
                "score": float(c["score"]),
                "fft_corr": c["fft_corr"],
                "dtw_distance": c["dtw_cost"],
                "dtw_normalized": c["dtw_normalized"],
                "offset_deg": float(c["offset_bin"] * bin_deg),
            }
        )

    confidence = _compute_confidence(matches)

    status = "OK"
    reason = "Match found"
    ok = True
    if confidence["best_score"] < min_corr:
        status = "LOW_CONFIDENCE"
        reason = f"Low correlation ({confidence['best_score']:.3f} < {min_corr})"
        ok = False
    elif confidence["ambiguous"] and confidence["score_gap"] < min_score_gap:
        status = "LOW_CONFIDENCE"
        reason = (
            f"Ambiguous top-2 score gap "
            f"({confidence['score_gap']:.4f} < {min_score_gap})"
        )
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
    """Refinement helper: DTW on streamed candidate row-views."""
    query_val, query_d1 = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T

    candidates = []
    for corr, _, vp_idx, offset, horizon_window in result_view:
        db_val, db_d1 = _feature_bundle(horizon_window)
        db_features = np.vstack([db_val, db_d1]).T
        dtw_cost, _ = fastdtw(
            query_features, db_features, radius=dtw_window, dist=euclidean
        )
        dtw_normalized = dtw_cost / max(len(query_features), 1)
        candidates.append(
            {
                "viewpoint_idx": int(vp_idx),
                "fft_corr": float(corr),
                "dtw_cost": float(dtw_cost),
                "dtw_normalized": float(dtw_normalized),
                "offset_bin": int(offset),
                "score": float(corr - 0.01 * dtw_normalized),
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
