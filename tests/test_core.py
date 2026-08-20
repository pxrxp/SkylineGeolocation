"""Core logic tests. No network, no GPU, no large data required."""

import json
import os
import tempfile
import numpy as np
from pathlib import Path

from src.region import Region, import_region
from src.matching import (
    match_query,
    _safe_zscore,
    _compute_confidence,
    saliency_weights,
    fit_affine_scale_offset,
    affine_scale_ok,
)
from src.query_profile import (
    is_profile_applicable,
    extract_elevation_profile,
    evaluate_skyline_quality,
    compute_column_keep_mask,
)
from src.segmentation import _compute_sky_diagnostics
from src.config import PipelineConfig


class TestRegion:
    def test_roundtrip(self):
        r = Region(-122.5, -122.0, 37.5, 38.0)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            r.save(f.name)
            fname = f.name
        try:
            r2 = import_region(fname)
            assert abs(r2.west_deg - (-122.5)) < 1e-10
            assert abs(r2.east_deg - (-122.0)) < 1e-10
            assert abs(r2.south_deg - 37.5) < 1e-10
            assert abs(r2.north_deg - 38.0) < 1e-10
        finally:
            os.unlink(fname)


class TestSafeZscore:
    def test_flat(self):
        x = np.ones(100) * 5.0
        z = _safe_zscore(x)
        assert z.shape == x.shape
        assert z.std() < 1e-12
        assert z.mean() < 1e-12

    def test_varied(self):
        x = np.sin(np.linspace(0, 10, 100)) * 5
        z = _safe_zscore(x)
        assert abs(z.std() - 1.0) < 0.01
        assert abs(z.mean()) < 0.01


class TestIsProfileApplicable:
    def test_empty(self):
        ok, msg = is_profile_applicable(np.array([]))
        assert not ok
        assert "Empty" in msg

    def test_flat(self):
        ok, msg = is_profile_applicable(np.zeros(100))
        assert not ok
        assert "flat" in msg.lower()

    def test_nan(self):
        ok, msg = is_profile_applicable(np.full(100, np.nan))
        assert not ok
        assert "NaN" in msg

    def test_good(self):
        ok, msg = is_profile_applicable(np.sin(np.linspace(0, 10, 100)) * 5)
        assert ok
        assert "Valid" in msg


class TestComputeSkyDiagnostics:
    def test_half_sky(self):
        mask = np.full((100, 100), 255, dtype=np.uint8)
        mask[:50, :] = 0
        d = _compute_sky_diagnostics(mask)
        assert abs(d["sky_ratio"] - 0.5) < 0.01
        assert d["boundary_coverage"] == 1.0
        assert d["top_connected"] is True

    def test_no_sky(self):
        mask = np.full((100, 100), 255, dtype=np.uint8)
        d = _compute_sky_diagnostics(mask)
        assert d["sky_ratio"] == 0.0
        assert d["num_components"] == 0


class TestMatchQuery:
    def test_empty_db(self):
        result = match_query(None, 0.25, np.zeros(100))
        assert result["status"] == "INVALID_INPUT"
        assert not result["ok"]

    def test_short_query(self):
        result = match_query(np.zeros((10, 360)), 0.25, np.ones(5))
        assert result["status"] == "INVALID_QUERY"
        assert not result["ok"]

    def test_nan_query(self):
        result = match_query(np.zeros((10, 360)), 0.25, np.full(100, np.nan))
        assert result["status"] == "INVALID_QUERY"
        assert not result["ok"]

    def test_runs(self):
        db = np.random.randn(20, 360)
        q = np.sin(np.linspace(0, 10, 360)) * 5
        result = match_query(db, 1.0, q, spatial_stride=2)
        assert "matches" in result
        assert "confidence" in result


class TestComputeConfidence:
    def test_empty(self):
        c = _compute_confidence([])
        assert c["ambiguous"]

    def test_gap(self):
        matches = [{"score": 0.5}, {"score": 0.1}]
        c = _compute_confidence(matches)
        assert abs(c["best_score"] - 0.5) < 0.01
        assert abs(c["score_gap"] - 0.4) < 0.01


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.dist_search_km == 30.0
        assert cfg.azim_num == 720
        assert cfg.bin_deg == 0.5
        assert cfg.min_corr == 0.30
        assert cfg.top_k == 5

    def test_override(self):
        cfg = PipelineConfig(dist_search_km=60.0, top_k=10)
        assert cfg.dist_search_km == 60.0
        assert cfg.top_k == 10


class TestSaliencyWeights:
    def test_flat_profile(self):
        w = saliency_weights(np.zeros(100))
        assert np.all(w == 1.0)

    def test_single_peak(self):
        p = np.zeros(200)
        p[100] = 10.0
        w = saliency_weights(p, alpha=2.0)
        assert w[100] > 1.0
        assert np.all(w[np.abs(np.arange(200) - 100) > 5] < w[100] + 1e-9)

    def test_alpha_zero(self):
        p = np.sin(np.linspace(0, 10, 200)) * 5
        w = saliency_weights(p, alpha=0.0)
        assert np.allclose(w, 1.0)

    def test_bounded_positive(self):
        p = np.sin(np.linspace(0, 10, 200)) * 5
        w = saliency_weights(p, alpha=2.0)
        assert w.min() >= 1.0
        assert np.isfinite(w).all()


class TestSkylineQualityGate:
    def test_flat_profile_rejected(self):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:, :] = 255
        passed, score, reason = evaluate_skyline_quality(img, mask, np.zeros(100))
        assert not passed
        assert reason == "FLAT_TERRAIN_NO_RELIEF"

    def test_empty_profile_rejected(self):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        passed, score, reason = evaluate_skyline_quality(img, mask, np.array([]))
        assert not passed
        assert reason == "EMPTY_PROFILE"


class TestAffineFit:
    def test_exact_affine_recovered(self):
        rng = np.random.default_rng(1)
        db = np.sin(np.linspace(0, 6, 720)) * 5 + 20
        offset = 123
        A_true, b_true = 1.2, -8.0
        profile = A_true * np.roll(db, -offset)[:175] + b_true
        A, b, rmse = fit_affine_scale_offset(db, profile, offset)
        assert abs(A - A_true) < 1e-6
        assert abs(b - b_true) < 1e-6
        assert rmse < 1e-6

    def test_scale_gate(self):
        assert affine_scale_ok(1.0)
        assert affine_scale_ok(0.6)
        assert affine_scale_ok(1.6)
        assert not affine_scale_ok(0.5)
        assert not affine_scale_ok(1.7)
        assert not affine_scale_ok(-0.5)


class TestColumnKeepMask:
    def test_flat_image_all_kept(self):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:, :] = 255
        keep, per_col = compute_column_keep_mask(img, mask, gradient_threshold=8.0)
        assert keep.shape == (64,)
        assert keep.dtype == bool

    def test_no_sky_columns_dropped(self):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)  # all terrain
        keep, per_col = compute_column_keep_mask(img, mask, gradient_threshold=8.0)
        assert not keep.any()
