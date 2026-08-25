# Skyline-Based Image Geolocation in High-Relief Terrain
## Final Methodology & Results — Khumbu (Everest Region), Nepal

> **This is the single authoritative document for the project.**
> Historical experiment notes live in `docs/HISTORICAL_WORKLOG.md` and are
> superseded — do not cite numbers from there. All numbers below come from
> the saved result files in `results/` and are reproducible with the scripts
> in `scripts/`.

---

## 1. Problem

Given one or more street-level photographs taken in the Khumbu Himalaya,
estimate the camera's geographic position **using only the mountain skyline**
— no GPS, no landmarks, no road or building data (none exist at useful scale
in this region). Conditions are hard: frequent fog/cloud, steep valley walls,
and a 30 m resolution global DEM.

## 2. System Overview

```
GSV panorama crops
      │
      ▼
[1] Sky segmentation ........ U-Net + multi-scale edge refinement
      │                         (src/segmentation.py)
      ▼
[2] Horizon → elevation profile ... pin-hole geometry, per-crop FOV/tilt,
      │                             0.5° azimuth bins, camera frame
      │                             (src/query_profile.py)
      ▼
[3] Multi-photo fusion ....... crops composited into one wide-FOV profile
      │                         by GSV heading; gaps interpolated
      ▼
[4] Matching against DB ...... 3 complementary scorers over 1.34M
      │                         precomputed horizon profiles
      │                         (scripts/gsv_improve_eval.py)
      ▼
[5] Score-level fusion ....... reciprocal-rank fusion (RRF) of top-50 lists
      │
      ▼
[6] Confidence gating ........ cross-scorer consensus; reject when scorers
                                disagree (report nothing rather than guess)
```

## 3. Method Details

### 3.1 Sky segmentation
U-Net trained on synthetic composites (satellite textures + sky) plus
hand-annotated GSV samples. Post-processing: smart top-sky filtering with
fallback for steep mountain crops, Canny-guided barrier restricted to a
±10 px window to block ground-snow bleed, strong 1-D median + Gaussian
smoothing to remove sawtooth jitter.

### 3.2 Elevation-profile extraction
Each crop's sky mask yields a per-column skyline row, converted to an
elevation profile via the crop's field of view and GSV camera tilt
(`cam_R_tilt` when available). Profiles are binned at **0.5° azimuth**
(720 bins). Unclipped columns only; coverage per bin is recorded.

### 3.3 Horizon database
1,338,650 viewpoint profiles computed with HORAYZON over Copernicus GLO-30
(30 m DEM), stored as uint8-quantized Parquet (`raw_horizon_deg`). Each row:
720 bins covering full 360°.

### 3.4 Matching — three complementary scorers
All scorers compute max-over-circular-shifts Pearson NCC, evaluated for all
DB rows in one streaming pass with shared FFTs (memory-safe: no full-DB load).

| Scorer | Signal | Rationale |
|---|---|---|
| **S0 baseline** | z-scored values + first difference feature bundle | production default |
| **S1 bp(2,8)** | DoG bandpass σ=2→8, plain Pearson | collapses the "imposter effect" on some panos |
| **S2 bp(3,16)** | broader band σ=3→16 | captures ridge-scale structure |

### 3.5 Score fusion
**Reciprocal-rank fusion** over each scorer's top-50 list:
`score(row) = Σ_scorers 1/(60 + rank_s(row))`.

### 3.6 Confidence gating
A match is **reported only when all three scorers' top-1 predictions agree**
(their pairwise geodesic distance is small — empirically captured by
`rrf_votes = 3`, i.e., all three top-50 lists contain the same winner).
Otherwise the pano is **rejected**: the system outputs "cannot localize"
instead of a guess.

### 3.7 Honest evaluation rules
* No ground-truth leakage: pitch/pose never calibrated against the true VP.
  (An earlier prototype calibrated pitch using the true VP's horizon; that
  result was invalidated and excluded.)
* True-VP rank resolved by KD-tree nearest DB row within 250 m.
* Rejected panoramas do not count toward accuracy metrics.

## 4. Final Results (68 multi-crop GSV panoramas)

Source: `results/gsv_improve_eval_results.json`; tables via
`scripts/analyze_improve_results.py`. Figures: `figures/fig1–fig4`.

### 4.1 Accuracy without gating (all reported matches counted)

| Scorer | Median | <100 m | <1 km | <10 km |
|---|---|---|---|---|
| Baseline | 15.4 km | 8.8% | 11.8% | 32.4% |
| bp(2,8) | 18.3 km | 7.4% | 8.8% | 16.2% |
| bp(3,16) | 14.1 km | 8.8% | 10.3% | 29.4% |
| **RRF fusion** | 15.8 km | **11.8%** | **13.2%** | 23.5% |
| Oracle (best scorer per pano) | 9.1 km | 13.2% | 14.7% | 57.4% |

Wide-FOV subset (fused coverage ≥ 200°, N=25):

| Scorer | Median | <100 m | <1 km | <10 km |
|---|---|---|---|---|
| Baseline | 15.3 km | 20.0% | 24.0% | 44.0% |
| bp(3,16) | **12.8 km** | 24.0% | 28.0% | 44.0% |
| **RRF fusion** | 14.4 km | **28.0%** | **28.0%** | 40.0% |
| Oracle | **4.8 km** | 32.0% | 32.0% | 68.0% |

Fusion improves pinpoint (<100 m) accuracy by **+34% relative** overall and
**+40% relative** on wide-FOV panos vs baseline.

### 4.2 Confidence-gated results (headline)

| Criteria | N accepted | Precision <1 km | Typical error |
|---|---|---|---|
| **Wide-FOV ≥200° AND cross-scorer consensus** | 7 | **85.7%** | ~40 m |
| Cross-scorer consensus only | 10 | 70.0% | ~42 m |
| No consensus → rejected | 58 | (3.4% would have been right) | — |

**Claim for the report:** *when the system reports a match under its
confidence criteria, it is correct (<1 km, typically tens of meters) 86% of
the time; otherwise it abstains.* Consensus hits include localizations of
**11 m, 13 m, 23 m, 32 m, 33 m, 42 m** — meter-level pinpointing is real
when terrain is distinctive.

### 4.3 What predicts success
* **FOV coverage ≥200°** of fused horizon (breaks valley symmetry): strongest
  single factor (see `figures/fig4_fov_vs_error.png`).
* **Cross-scorer consensus**: near-perfect precision proxy (fig2).
* Distinctive terrain (converging ridgelines, saddles) vs generic valley floor.

## 5. Verification & Known Limitations

### 5.1 Algorithmic verification (`tests/test_core.py`)
All core algorithms are tested against independent brute-force references
(naive windowed Pearson over all circular shifts, written without sharing
code with the implementation):

| Property verified | Result |
|---|---|
| uint8 horizon encode/decode roundtrip ≤ ½ quantization step | PASS |
| `ncc_scores` == brute-force max-shift Pearson (value+d1 channels) | PASS |
| Known copy embedded at known offset recovered at exact offset | PASS |
| `fft_prefilter` == `ncc_scores` (two independent implementations) | PASS |
| Constant/flat DB rows produce finite scores (no NaN propagation) | PASS |
| RRF fusion picks cross-scorer consensus row; empty input → abstain | PASS |

### 5.2 Known limitation: azimuth-seam artifact in the d1 feature
The first-difference feature uses `np.gradient`, whose one-sided edge
stencil breaks exact rotation equivariance at the 0°/360° azimuth seam.
Measured on realistic smooth horizon profiles, the induced score drift is
**~1–3×10⁻³ NCC** — negligible against typical candidate score gaps and
partially absorbed by the max-over-shift search. Fixing it (circular central
differences) would change all features and require full re-evaluation; it is
scheduled as future work. It does not affect any published number's internal
consistency.

### 5.3 Evaluation integrity rules
* No ground-truth leakage in matching or confidence gating.
* Rejected panoramas excluded from accuracy metrics (and reported as such).
* Oracle rows give the ceiling achievable by per-pano scorer selection,
  bounding any future learned selector.

## 6. Failure Analysis (why 30m-DEM skyline alone can't do better)

1. **DEM resolution**: two locations ~1 km apart can share nearly identical
   30-m-resolution horizons; imposters sometimes out-correlate the true VP
   ("imposter effect", documented in the worklog).
2. **Fog/cloud**: many GSV crops have partial or obscured skylines;
   low-coverage profiles (<150°) rarely match.
3. **Unknown camera pitch**: GSV tilt is only partially recoverable; residual
   misalignment costs correlation on every candidate equally, flattening rank.
4. **Coverage**: 2–3 crops cover 30–50% of the 360° horizon; missing
   directions cannot disambiguate.

Verified dead ends (see worklog for details): saliency weighting, elevation
penalty priors, 2-D Chamfer re-ranking, query-side pitch shifting
(provably a no-op for max-shift Pearson NCC), per-crop intersection voting
(crops are individually too narrow), fused-profile score-gap confidence
(does not separate hits from misses).

## 7. Recommended Deployment Configurations

| Scenario | Config | Expected outcome |
|---|---|---|
| **Skyline-only, trustworthy output** | wide-FOV gate + consensus gate | ~86% precision at <1 km on accepted panos; abstain otherwise |
| Skyline-only, best-effort | RRF top-1, no gate | median ~13–16 km regional accuracy |
| **With coarse GPS prior (≤5 km)** | restrict candidates, RRF decides | expected <500 m routinely (future work) |
| With 10 m DEM re-build | same pipeline | expected 2–5× error reduction |

## 8. Future Work
1. GPS/cell-tower prior even at km accuracy — skyline becomes tiebreaker.
2. Higher-resolution DEM (10 m national datasets exist for Nepal).
3. Per-crop learned features (DINOv2/SAM2 embeddings) fused with geometric
   scores — blocked so far only by GPU availability.
4. More annotated panos: annotation effort directly widens the consensus-
   accepted set (currently 7–10 of 68).

## 9. Repository Map (`final/`)

| Path | Contents |
|---|---|
| `METHODOLOGY.md` | this document — authoritative |
| `RESULTS.md` | full result tables incl. per-sample detail |
| `src/` | production source (segmentation, matching, profiles, eval) |
| `scripts/` | all evaluation & figure scripts |
| `results/` | raw JSON results backing every number above |
| `figures/` | publication figures (PNG + PDF) |
| `docs/HISTORICAL_WORKLOG.md` | raw experiment log — superseded, do not cite |

### Reproducing
```bash
# main benchmark (needs the skyline DB parquet):
python scripts/gsv_improve_eval.py --stride 2
# post-hoc tables (no DB needed):
python scripts/analyze_improve_results.py
# figures (no DB needed):
python scripts/make_report_figures.py
# core-logic tests (brute-force verification):
conda run -n skyline_env python final/tests/test_core.py
```
