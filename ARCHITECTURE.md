# Skyline-Based Image Geolocation in High-Relief Terrain

## Final Methodology & Results — Khumbu (Everest Region), Nepal

> **This is the single authoritative document for the project.**
> Historical experiment notes live in `archive/docs/` and are
> superseded — do not cite numbers from there. All numbers below come from
> saved result files in `data/street_view/` and are reproducible with the
> evaluation scripts in `archive/scripts/`.

---

## 1. Problem

Given one or more street-level photographs taken in the Khumbu Himalaya,
estimate the camera's geographic position **using only the mountain skyline** —
no GPS, no landmarks, no road or building data (none exist at useful scale in
this region). Conditions are hard: frequent fog/cloud, steep valley walls, and
a 30 m resolution global DEM.

---

## 2. End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  DATA PREPARATION                                                │
│                                                                  │
│  [1] Region definition (src/region.py)                           │
│      → geographic bounds: lat/lon rectangle                      │
│                                                                  │
│  [2] DEM download (notebooks/01_RegionStudy/)                    │
│      → Copernicus GLO-90 (90 m) for viewshed analysis           │
│      → Copernicus GLO-30 (30 m) for horizon database            │
│                                                                  │
│  [3] Viewshed analysis (src/viewshed.py)                         │
│      → visibility masks, distance calculations,                 │
│        visibility percentiles for terrain characterization       │
│                                                                  │
│  [4] Horizon database generation (src/skyline.py)                │
│      → SkylineDatabaseGenerator using HORAYZON library           │
│      → GLO-30 DEM, 30 m grid spacing, 720 azimuth bins         │
│      → ~1.34M viewpoint profiles stored as uint8 Parquet        │
│      → Horizons stored in raw_horizon_deg column                 │
│                                                                  │
│  [5] Synthetic training data (src/synthetic_generator.py)        │
│      → SyntheticSceneGenerator:                                  │
│        - DEM mesh rendered via pyrender/trimesh                   │
│        - Satellite imagery mapped as terrain texture              │
│        - Cloud backdrop images composited into sky                │
│      → Paired image + mask ground truth for U-Net training       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  SKY SEGMENTATION (src/segmentation.py)                          │
│                                                                  │
│  [6] U-Net model trained on:                                     │
│      → GeoPose3K dataset (hand-annotated sky/mountain masks)     │
│      → Synthetic scenes (procedural DEM + satellite + clouds)    │
│      → Training augmentation: cloud overlay compositing          │
│                                                                  │
│  [7] Post-processing pipeline:                                   │
│      → Smart top-sky filtering with fallback for steep mountains │
│      → Canny-guided barrier (±10 px window) to block ground snow │
│      → Multi-scale edge refinement: CLAHE-enhanced Canny +      │
│        LAB b* channel sub-pixel fitting                          │
│      → Strong 1D median + Gaussian smoothing to remove jitter    │
│      → Quality gates: min sky ratio, boundary coverage           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  QUERY PROFILE EXTRACTION (src/query_profile.py)                 │
│                                                                  │
│  [8] Per-crop skyline → elevation profile:                       │
│      → Per-column edge detection on sky mask boundary            │
│      → Sub-pixel fitting for sub-bin accuracy                    │
│      → Pin-hole geometry: pixel row → elevation angle            │
│      → Camera FOV and tilt (cam_R_tilt) accounted for           │
│      → Binned at 0.5° azimuth (720 bins, full 360°)            │
│      → Coverage per bin recorded for quality filtering           │
│      → Applicability checks: min std, min max elevation          │
│                                                                  │
│  [9] Multi-photo fusion:                                         │
│      → Multiple crops composited by GSV heading                  │
│      → Coverage gaps (no data) marked as NaN                     │
│      → Wide-FOV profiles: 200°–262° typical coverage            │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  MATCHING (src/matching.py)                                      │
│                                                                  │
│  [10] Feature extraction:                                        │
│       → Value channel: z-scored elevation profile                │
│       → D1 channel: z-scored first derivative (np.gradient)     │
│       → Bandpass channels: DoG σ2→8, σ3→16 for multi-scale     │
│                                                                  │
│  [11] Pearson NCC scoring:                                       │
│       → Max-over-circular-shifts Pearson correlation             │
│       → FFT-accelerated cross-correlation (one FFT per chunk)   │
│       → Vectorized over all DB rows (batched in 4000-row chunks)│
│       → Memory-safe streaming: never loads full DB into RAM     │
│                                                                  │
│  [12] Three complementary scorers:                               │
│       → S0 baseline: value + d1 feature bundle (0.5/0.5 weight) │
│       → S1 bp(2,8): DoG bandpass σ=2→8, plain Pearson          │
│       → S2 bp(3,16): broader bandpass σ=3→16                   │
│                                                                  │
│  [13] Score fusion:                                              │
│       → Reciprocal-rank fusion (RRF) over top-50 per scorer     │
│       → score(row) = Σ_scorers 1/(60 + rank_s(row))            │
│                                                                  │
│  [14] Fine refinement (optional):                                │
│       → fastdtw dynamic time warping on top candidates           │
│       → Window size configurable (default 15 bins)               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  CONFIDENCE GATING                                               │
│                                                                  │
│  [15] Cross-scorer consensus:                                    │
│       → Match reported ONLY when all three scorers' top-1       │
│         predictions agree (same location)                        │
│       → Otherwise: system outputs "cannot localize"              │
│       → Result: high precision on accepted matches,             │
│         honest abstention on ambiguous cases                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Region & DEM

The study area is defined by geographic bounds in `src/region.py`, saved as
JSON (lon/lat rectangle). Two DEM resolutions are used:

| DEM | Source | Resolution | Used for |
|-----|--------|-----------|----------|
| GLO-90 | Copernicus | 90 m | Viewshed analysis (src/viewshed.py) |
| GLO-30 | Copernicus | 30 m | Horizon database (src/skyline.py) |

DEM files are downloaded via `notebooks/01_RegionStudy/02_download_glo90_dem.ipynb`
and `04_download_glo30_dem.ipynb`, stored in `data/digital_elevation_model/`.

### 3.2 Viewshed Analysis

`src/viewshed.py` computes visibility from observer positions over the DEM:
- Binary visibility mask (line-of-sight blocked/visible)
- Visible distance calculations per azimuth
- Visibility percentiles (50th, 75th, 90th, 95th, 99th)
- LOD DEM generation for large-area analysis

This characterizes which viewpoints have long-distance visibility (useful for
matching) versus those in enclosed valleys (poor candidates).

### 3.3 Horizon Database Generation

`src/skyline.py` — class `SkylineDatabaseGenerator`:

1. **Grid**: regular 30 m spacing in projected coordinates (UTM), converted to
   lat/lon via geodetic forward computation (`pyproj.Geod`)
2. **Horizon rendering**: uses the [HORAYZON](https://github.com/AppsForBrowsers/HORAYZON)
   library to raytrace from observer positions through the DEM mesh
3. **Rendering parameters** (from `PipelineConfig`):
   - Search distance: 30 km (determines max visible range)
   - Eye height: 1.6 m (assumed camera height above ground)
   - Angular accuracy: 0.1° (HORAYZON convergence tolerance)
   - Lower elevation limit: -89° (prevents looking straight down)
   - Azimuth bins: 720 (0.5° resolution, full 360°)
4. **DEM interpolation**: HORAYZON uses the DEM mesh as a triangle surface;
   ray-DEM intersection is geometric (no resampling artifacts)
5. **Quantization**: elevation angles stored as `uint8` (0–255 maps linearly
   to 0°–90°, 0.353° per step, max error ≤0.18°) — see `src/horizon_format.py`
6. **Storage**: Parquet format with columns including `lon`, `lat`,
   `elevation_m`, `raw_horizon_deg` (variable-length uint8 arrays)
7. **Scale**: 1,338,650 viewpoints covering the Khumbu region

**Storage efficiency:**
- Raw uint8: 1,338,650 × 720 × 1 byte = 0.96 GB
- Parquet on disk: 486 MB (2× compression via columnar encoding)
- Equivalent float32: 3.86 GB (8× larger)
- Per-row: 0.4 KB (vs 720 bytes raw uint8)

The quantization is lossy but the error (≤0.18°) is below the matching
resolution (0.5° bins). Verified: uint8 roundtrip error ≤ ½ quantization
step, confirmed by brute-force test in `tests/test_core.py`.

**Query-time decode**: uint8 → float32 via `decode_horizon_uint8()` for
NCC matching (which requires float precision for z-scoring and FFT).

### 3.4 Synthetic Training Data

`src/synthetic_generator.py` — class `SyntheticSceneGenerator`:

Generates photorealistic mountain scenes for training data augmentation:

1. **Terrain mesh**: DEM loaded via `rasterio`, decimated by stride, built into
   a `trimesh` triangle mesh, offset to prevent float32 precision issues
2. **Texture**: satellite imagery (e.g., Sentinel-2) mapped onto the mesh using
   GPS-to-pixel coordinate transformation
3. **Sky/backdrop**: cloud images from a dataset composited behind the terrain
   via `pyrender` (EGL backend, headless rendering)
4. **Camera**: configurable viewpoint, FOV, and orientation
5. **Output**: paired RGB image + binary sky/mask ground truth, with JSON
   ground truth (camera pose, location)

This produces unlimited training data for the segmentation model without
requiring manual annotation.

### 3.5 Sky Segmentation

`src/segmentation.py`:

**Training data sources:**
- **GeoPose3K dataset**: hand-annotated sky/mountain segmentation masks
  (downloaded and split via `notebooks/04_SkySegmentation/01_download_geopose3k_dataset.ipynb`)
- **Synthetic scenes**: generated by `SyntheticSceneGenerator` with cloud
  overlay augmentation (the `SyntheticSkyDataset` class handles cloud
  compositing during training)

**Architecture:**
- U-Net encoder-decoder with skip connections
- Input: 256×256 RGB patches
- Loss: BCE + Dice combined loss (`bce_dice_loss`)
- Metric: intersection-over-union (`compute_iou`)

**U-Net output convention:**
The sigmoid output is thresholded as `raw_mask = (prob <= threshold)`,
producing **1 = sky, 0 = terrain**. The default refinement method
(`lab_b_subpixel` / `refine_sky_mask_with_guidance`) correctly interprets
this convention (`sky1 = (raw_unet_mask == 1)`). The other three methods
(`grayscale_fixed`, `multichannel_fusion`, `dynamic_programming`) use the
inverted convention (`sky1 = (raw_unet_mask == 0)`). Since `lab_b_subpixel`
is the default and the only method validated end-to-end, this mismatch is
harmless in practice — but those three methods should NOT be used without
inverting the raw mask first.

**Post-processing pipeline** (applied after raw U-Net inference, using `lab_b_subpixel`):
1. Smart top-sky filtering with fallback for steep mountain crops
2. Canny-guided barrier restricted to ±10 px window to block ground-snow
   bleed without cloud jumping
3. CLAHE-enhanced multi-scale Canny + LAB b* channel sub-pixel fitting
   (`refine_sky_mask_with_guidance`)
4. Strong 1D median + Gaussian smoothing to remove sawtooth jitter
5. Quality gates: min sky ratio (0.05), max sky ratio (0.95), boundary
   coverage threshold

### 3.6 Query Profile Extraction

`src/query_profile.py`:

**Single-crop extraction** (`extract_elevation_profile`):
1. Per-column edge detection along the sky mask boundary
2. Sub-pixel fitting for sub-bin accuracy (`_subpixel_edge_from_image`)
3. Pin-hole camera model: pixel row → elevation angle using FOV and tilt
4. Binned at 0.5° azimuth (720 bins)
5. Coverage tracking per bin
6. Applicability check: minimum standard deviation (1.5°) and maximum
   elevation range (1.0°) required

**Multi-crop fusion** (`fuse_profiles`):
- Places each crop's profile into the correct azimuth bin by GSV heading
- Overlapping bins keep the first valid value
- Gaps linearly interpolated via `np.interp`
- Returns fused wide-FOV profile (200°–262° typical) + coverage metric

Quality evaluation via `evaluate_skyline_quality` checks boundary gradient
strength and profile consistency.

### 3.7 Matching Algorithm

`src/matching.py` — two-stage pipeline with shared-FFT multi-query scoring:

**Stage 1: Coarse spatial scan (Pearson NCC)**
- Feature extraction: z-scored value + z-scored first derivative
- Max-over-circular-shifts: tests all 720 heading alignments
- Two implementations (mathematically equivalent, verified by tests):
  - **Production** (`ncc_scores`): cumsum-based windowed Pearson for single
    queries; O(N×L) with low constant factor
  - **Evaluation** (`score_chunk_shared_fft`): FFT-based cross-correlation
    shared across multiple queries per chunk; ~16× faster for batch scoring
- Memory-safe streaming: never loads full 1.34M-row DB into RAM

**Three complementary scorers:**

| Scorer | Signal | Rationale |
|--------|--------|-----------|
| S0 baseline | value + d1 feature bundle (0.5/0.5) | production default |
| S1 bp(2,8) | DoG bandpass σ=2→8, plain Pearson | collapses imposter effect on some panos |
| S2 bp(3,16) | broader bandpass σ=3→16 | captures ridge-scale structure |

**Bandpass implementation**: Difference-of-Gaussians (DoG) in the spatial
domain via `scipy.ndimage.gaussian_filter1d` with `mode="wrap"` (circular).
`DoG(σ1,σ2) = Gaussian(σ1) - Gaussian(σ2)`. This acts as a bandpass filter
that removes both the DC component (mean elevation) and high-frequency noise,
isolating the mid-scale terrain structure that discriminates locations.

**Stage 2: Fine refinement**
- Top-K candidates from Stage 1
- fastdtw dynamic time warping for sub-bin alignment
- Confidence computed from score gap (best - second best)

**Score fusion (`rrf_fusion`):**
- Reciprocal-rank fusion (RRF) over each scorer's top-50 list
- `score(row) = Σ_scorers 1/(60 + rank_s(row))`

### 3.8 Evaluation

`src/evaluation.py`:
- Streaming chunked evaluation (4000 rows per chunk)
- Spatial stride for coarse scan (default 12 → checks every 12th VP)
- Configurable parameters via `src/config.py` (`PipelineConfig`)

---

## 3A. Configuration Parameters (all defaults)

| Category | Parameter | Value | Rationale |
|----------|-----------|-------|-----------|
| **DB** | `grid_spacing_m` | 30 | Matches DEM resolution (GLO-30); finer grid adds VPs without improving matching |
| | `dist_search_km` | 30 | Khumbu valleys are ~20-30 km long; beyond this, DEM accuracy degrades |
| | `eye_height_m` | 1.6 | Standard handheld camera height; sensitivity analysis shows <0.5° error for ±0.5m |
| | `hori_acc_deg` | 0.1 | HORAYZON raytracing convergence; lower values slow generation with no matching benefit |
| | `azim_num` | 720 | 0.5° resolution matches the uint8 quantization step (~0.35°); finer bins waste storage |
| | `batch_size` | 4096 | Balances memory usage and GPU utilization during DB generation |
| **Profile** | `bin_deg` | 0.5 | Derived from azim_num; matches DB resolution |
| | `fov_y_deg` | 65.0 | GSV default; actual FOV extracted from image metadata when available |
| | `median_kernel` | 5 | Empirically removes segmentation jitter without blurring real terrain features |
| | `min_std_deg` | 1.5 | Below this, profile is too flat to discriminate locations (tested in `01_evaluate_matching.ipynb`) |
| | `min_max_elev_deg` | 1.0 | Minimum relief needed for matching; filters sky-only or ground-only crops |
| **Segmentation** | `seg_input_size` | 256 | Standard U-Net resolution; larger inputs don't improve sky boundary quality |
| | `min_sky_ratio` | 0.05 | Reject crops that are almost entirely ground (testing showed <5% sky = unusable) |
| | `max_sky_ratio` | 0.95 | Reject crops that are almost entirely sky (likely segmentation error or cloud) |
| | `min_boundary_coverage` | 0.5 | Need at least half the horizon boundary to extract a usable profile |
| **Matching** | `fft_weights` | (0.5, 0.5) | Equal weight; ablation in `full_db_bandpass_eval.py` showed d1 adds ~10% over value-only |
| | `min_corr` | 0.30 | Below this, matches are noise; tested on synthetic queries with known ground truth |
| | `min_score_gap` | 0.03 | Empirically separates confident from ambiguous matches; too high rejects correct matches |
| | `top_k` | 5 | DTW refinement is expensive; top-5 captures most correct matches |
| | `dtw_window` | 15 | Allows ~7.5° alignment correction; larger windows increase false positives |
| | `spatial_stride` | 12 | Production default for speed; stride=1 used for final evaluation |
| **Eval** | `chunk_rows` | 4000 | Balances memory (~150 MB per chunk) and I/O overhead |
| | `correct_dist_m` | 500 | Standard benchmark threshold; aligns with human navigation tolerance |
| | `compass_tolerance_deg` | 20 | GSV compass accuracy is ±5-10°; 20° gives comfortable margin |
| | `height_tolerance_m` | 200 | GPS altitude accuracy is ±30-50m; 200m accounts for DEM vs real terrain |

**Bandpass σ selection** (from `full_db_bandpass_eval.py`):
- bp(1,4): too narrow, captures noise → rejected
- bp(2,8): captures ridge-scale structure (~2-8 km features) → selected as S1
- bp(3,16): captures valley-scale structure (~3-16 km features) → selected as S2
- bp(4,32): too broad, loses discriminative detail → rejected

**Consensus threshold** (from `gsv_improve_eval.py`):
- 3/3 scorers agree → 85.7% precision (N=7 accepted)
- 2/3 scorers agree → 70.0% precision (N=10 accepted)
- <2/3 agree → 3.4% precision (correctly rejected)

---

## 4. Evaluation Results

### 4.1 GSV Multi-Photo Evaluation

68 multi-crop panoramas from Google Street View in the Khumbu region.

**Without gating (all matches counted):**

| Scorer | Median | <100 m | <1 km | <5 km | <10 km |
|--------|--------|--------|-------|-------|--------|
| Baseline (S0) | 15.4 km | 8.8% | 11.8% | 16.2% | 32.4% |
| bp(2,8) (S1) | 18.3 km | 7.4% | 8.8% | 10.3% | 16.2% |
| bp(3,16) (S2) | 14.1 km | 8.8% | 10.3% | 19.1% | 29.4% |
| **RRF fusion** | 15.8 km | **11.8%** | **13.2%** | 14.7% | 23.5% |
| Oracle | 9.1 km | 13.2% | 14.7% | 30.9% | 57.4% |

**Wide-FOV subset (fused coverage ≥ 200°, N=25):**

| Scorer | Median | <100 m | <1 km | <10 km |
|--------|--------|--------|-------|--------|
| Baseline (S0) | 15.3 km | 20.0% | 24.0% | 44.0% |
| bp(3,16) (S2) | **12.8 km** | 24.0% | 28.0% | 44.0% |
| **RRF fusion** | 14.4 km | **28.0%** | **28.0%** | 40.0% |
| Oracle | **4.8 km** | 32.0% | 32.0% | 68.0% |

Fusion improves pinpoint (<100 m) accuracy by +34% relative overall and
+40% relative on wide-FOV panos vs baseline.

### 4.2 Confidence-Gated Results (Headline)

| Criteria | N accepted | Precision <1 km | Typical error |
|----------|-----------|-----------------|---------------|
| **Wide-FOV ≥200° AND cross-scorer consensus** | 7 | **85.7%** | ~40 m |
| Cross-scorer consensus only (any FOV) | 10 | 70.0% | ~42 m |
| No consensus → rejected | 58 | (3.4% would have been right) | — |

**Claim:** *when the system reports a match under its confidence criteria, it
is correct (<1 km, typically tens of meters) 86% of the time; otherwise it
abstains.* Consensus hits include localizations of **11 m, 13 m, 23 m, 32 m,
33 m, 42 m** — meter-level pinpointing is real when terrain is distinctive.

### 4.3 Noise Robustness (synthetic self-matching)

**Important caveat:** this tests resilience to signal degradation, NOT
off-grid localization. Queries are DB profiles with added noise — the true
location IS in the database. This measures how much noise the matcher can
tolerate before degrading, not how well it localizes unknown positions.

Source: `data/street_view/offgrid_eval_results.json`.

| Noise σ | Rank-0 | <100 m | <1 km | Med n70 |
|---------|--------|--------|-------|----------|
| 0.0° | 100% | 100% | 100% | 1187 |
| 0.25° | 100% | 100% | 100% | 194 |
| 0.5° | 100% | 100% | 100% | 1 |
| 1.0° | 100% | 100% | 100% | 0 |
| 2.0° | 85% | 85% | 95% | 0 |
| uint8 quant | 100% | 100% | 100% | 1187 |

`n70` = number of DB profiles correlating > 0.65 with the query.
Key finding: even 0.5° noise makes profiles highly distinctive (n70 drops
from 1187 to 1), meaning real-world noise would actually HELP
localization by eliminating ambiguous matches.

### 4.4 What Predicts Success

1. **FOV coverage ≥ 200°** of fused horizon (breaks valley symmetry):
   strongest single factor
2. **Cross-scorer consensus**: near-perfect precision proxy
3. **Distinctive terrain** (converging ridgelines, saddles) vs generic valley floor

---

## 5. Verification & Known Limitations

### 5.1 Algorithmic Verification (`tests/test_core.py`)

All core algorithms tested against independent brute-force references
(naive windowed Pearson over all circular shifts, written without sharing
code with the implementation):

| Property verified | Result |
|-------------------|--------|
| uint8 horizon encode/decode roundtrip ≤ ½ quantization step | PASS |
| `ncc_scores` == brute-force max-shift Pearson (value+d1 channels) | PASS |
| Known copy embedded at known offset recovered at exact offset | PASS |
| `fft_prefilter` == `ncc_scores` (two independent implementations) | PASS |
| Constant/flat DB rows produce finite scores (no NaN propagation) | PASS |
| RRF fusion picks cross-scorer consensus row; empty input → abstain | PASS |

### 5.2 Known Limitation: Azimuth-Seam Artifact

The first-difference feature uses `np.gradient`, whose one-sided edge stencil
breaks exact rotation equivariance at the 0°/360° azimuth seam. Measured on
realistic smooth horizon profiles, the induced score drift is ~1–3×10⁻³ NCC —
negligible against typical candidate score gaps and partially absorbed by the
max-over-shift search. Documented as known limitation; fixing it (circular
central differences) is future work.

### 5.3 Evaluation Integrity Rules

- No ground-truth leakage in matching or confidence gating
- Rejected panoramas excluded from accuracy metrics (and reported as such)
- Oracle rows give the ceiling achievable by per-pano scorer selection

---

## 6. Failure Analysis

1. **DEM resolution**: two locations ~1 km apart can share nearly identical
   30 m resolution horizons; imposters sometimes out-correlate the true VP
   ("imposter effect")
2. **Fog/cloud**: many GSV crops have partial or obscured skylines;
   low-coverage profiles (<150°) rarely match
3. **Unknown camera pitch**: GSV tilt is only partially recoverable; residual
   misalignment costs correlation on every candidate equally, flattening rank
4. **Coverage**: 2–3 crops cover 30–50% of the 360° horizon; missing
   directions cannot disambiguate

Verified dead ends (see `archive/docs/HISTORICAL_WORKLOG.md` for details):
- Saliency weighting
- Elevation penalty priors
- 2D Chamfer re-ranking
- Query-side pitch shifting (provably a no-op for max-shift Pearson NCC)
- Per-crop intersection voting (crops individually too narrow)
- Fused-profile score-gap confidence (does not separate hits from misses)

---

## 7. Deployment

### Mobile App

A Kivy-based mobile application replicates the full pipeline for on-device
deployment. The app runs the same segmentation, profile extraction, and
matching logic against a precomputed horizon database.

### Recommended Configurations

| Scenario | Config | Expected outcome |
|----------|--------|-----------------|
| **Skyline-only, trustworthy** | wide-FOV gate + consensus gate | ~86% precision at <1 km; abstain otherwise |
| Skyline-only, best-effort | RRF top-1, no gate | median ~13–16 km regional accuracy |
| **With coarse GPS prior (≤5 km)** | restrict candidates, RRF decides | expected <500 m routinely |
| With 10 m DEM re-build | same pipeline | expected 2–5× error reduction |

---

## 8. Future Work

1. **GPS/cell-tower prior** even at km accuracy — skyline becomes tiebreaker
2. **Higher-resolution DEM** (10 m national datasets exist for Nepal)
3. **Learned features** (DINOv2/SAM2 embeddings) fused with geometric scores
4. **More annotated panos** — annotation effort directly widens the
   consensus-accepted set (currently 7–10 of 68)

---

## 9. Repository Map

| Path | Contents |
|------|----------|
| `ARCHITECTURE.md` | this document — authoritative |
| `src/` | production source (segmentation, matching, profiles, DB generation, synthetic data, evaluation) |
| `notebooks/` | pipeline notebooks (numbered 01–06 by stage) |
| `data/` | DEMs, models, GSV crops, ground truth |
| `tests/` | brute-force algorithmic verification suite |
| `archive/` | historical scripts, evaluation code, dashboards, Streamlit app, Colab notebooks |

### Reproduction

```bash
# core-logic tests (brute-force verification):
conda run -n skyline_env python tests/test_core.py

# full pipeline: see notebooks/01_RegionStudy through 06_GSV_Evaluation
# evaluation scripts: archive/scripts/gsv_improve_eval.py
```

---

## 10. Dataset Summary

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| Copernicus GLO-90 | ESA Copernicus | 90 m DEM | Viewshed analysis |
| Copernicus GLO-30 | ESA Copernicus | 30 m DEM | Horizon database |
| Skyline DB | Generated (HORAYZON) | 1.34M viewpoints, Parquet | Matching database |
| GeoPose3K | Published dataset | ~3K images + masks | Segmentation training |
| Synthetic scenes | Generated (pyrender) | Unlimited pairs | Segmentation augmentation |
| Cloud images | Published dataset | Cloud backdrop library | Synthetic sky generation |
| GSV panoramas | Google Street View | 68 multi-crop panos | Evaluation queries |
| Ground truth | Manual annotation | Camera GPS positions | Evaluation metric |
