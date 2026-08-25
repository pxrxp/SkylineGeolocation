# Skyline-Based Image Geolocation in High-Relief Terrain

**Khumbu (Everest region), Nepal — mountain-skyline-only localization from street-level photos.**

Given one or more photos, estimate the camera's geographic position using
only the mountain skyline: no GPS, no landmarks, no roads or buildings
(none exist at useful scale in this terrain). This folder is the complete,
self-contained deliverable: methodology, results, source code, evaluation
scripts, and figures.

---

## Headline results

| Claim | Number |
|---|---|
| **Precision when the system reports a match** (wide-FOV ≥200° + cross-scorer consensus) | **85.7% correct at <1 km, median ≈ 40 m** (6/7 hits < 100 m; best 11 m) |
| Accuracy when it abstains | rejected panos would have been only 3–6% right — gating removes noise |
| Best ungated matcher (RRF fusion of 3 scorers) | median 15.8 km overall; **12.8 km / 28% <100 m** on wide-FOV subset |
| vs production baseline | +34% relative pinpoint (<100 m) accuracy overall, **+40%** on wide-FOV |
| Oracle ceiling (best scorer per pano, wide-FOV) | 4.8 km median, 32% < 1 km |

**How to read this:** skyline matching alone identifies the correct *region*
(10–15 km) for most photos and *pinpoints* (tens of meters) a subset with
distinctive terrain. The system detects when it can pinpoint (scorer
consensus) and reports nothing otherwise — that trade is deliberate and is
the core result.

## Quick start

```bash
conda activate skyline_env   # or: conda run -n skyline_env <cmd>

# tables + confidence gates from saved results (instant, no DB needed)
python scripts/analyze_improve_results.py

# publication figures fig1–fig4 (~30 s, no DB needed)
python scripts/make_report_figures.py

# core-algorithm verification against brute-force references (11 tests)
python tests/test_core.py

# full benchmark (needs skyline_db.parquet, ~30 min)
python scripts/gsv_improve_eval.py --stride 2
```

All scripts work whether run from inside `final/` or from the repository
root — paths are resolved automatically.

## How this folder is organized

| Path | What it is |
|---|---|
| [`METHODOLOGY.md`](METHODOLOGY.md) | **Start here.** Authoritative methodology + results + verification + limitations |
| [`RESULTS.md`](RESULTS.md) | Full result tables incl. per-sample detail for all 68 panoramas |
| `src/` | Production source: sky segmentation, profile extraction, matching, eval |
| `scripts/` | Evaluation & figure scripts (see quick start) |
| `results/` | Raw JSON outputs backing every number quoted in the docs |
| `figures/` | Publication figures (PNG + PDF): error CDFs, confidence gates, baseline-vs-fusion scatter, coverage-vs-error |
| `tests/` | Logic tests comparing optimized matchers against independent brute-force references |
| `docs/HISTORICAL_WORKLOG.md` | Raw experiment log — **superseded, do not cite numbers from it** |

## Method in one paragraph

U-Net sky segmentation → geometric horizon→elevation profiles at 0.5°
azimuth bins → multi-crop fusion into one wide-field-of-view profile →
three complementary Pearson-NCC scorers over 1.34M DEM-simulated horizon
profiles (baseline value+gradient, DoG bandpass bp(2,8), bp(3,16)) →
reciprocal-rank score fusion → **confidence gate: report only when all
three scorers independently agree on the same location, otherwise abstain**.

## Key facts & integrity notes

* Database: 1,338,650 viewpoint horizons (HORAYZON × Copernicus GLO-30, 30 m DEM).
* Queries: 68 multi-crop GSV panoramas, Khumbu region.
* No ground-truth leakage: an early prototype calibrated camera pitch using
  the true viewpoint's horizon; that result was invalidated and excluded.
* Known limitation (documented, quantified at ~1–3×10⁻³ NCC): azimuth-seam
  artifact in the gradient feature — see METHODOLOGY §5.2.
* Fundamental limit: at 30 m DEM resolution many distinct locations share
  near-identical horizon shapes ("imposter effect"); sub-km precision for
  all photos requires a GPS prior or higher-resolution DEM (METHODOLOGY §6–8).
