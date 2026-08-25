# Skyline-Based Image Geolocation

Mountain skyline matching for camera localization in the Khumbu Himalaya
(Everest region), Nepal. No GPS, no landmarks — only the mountain horizon
visible in street-level photographs.

## Repository Structure

```
src/              Production library (17 modules)
notebooks/        Pipeline notebooks (numbered 01–06 by stage)
data/             DEMs, models, GSV crops, ground truth (not in git — 7+ GB)
tests/            Algorithmic verification suite (11/11 passing)
archive/          Evaluation scripts, dashboards, Streamlit app, Colab notebooks
ARCHITECTURE.md   Full methodology, pipeline diagram, results, verification
```

## Pipeline Overview

1. **Region & DEM** → Copernicus GLO-30 (30 m) over Khumbu
2. **Horizon DB** → 1.34M viewpoints via HORAYZON, stored as uint8 Parquet
3. **Synthetic data** → DEM mesh + satellite texture + cloud backdrops (pyrender)
4. **Sky segmentation** → U-Net trained on GeoPose3K + synthetic data
5. **Profile extraction** → Sky mask → 720-bin elevation profile per crop
6. **Multi-crop fusion** → Wide-FOV profiles (200°–262°)
7. **3-scorer matching** → Baseline NCC + bandpass σ(2,8) + bandpass σ(3,16)
8. **RRF fusion** → Reciprocal-rank fusion of top-50 lists
9. **Confidence gating** → Report only when all scorers agree; abstain otherwise

See `ARCHITECTURE.md` for full details, results, and verification.

## Headline Results

- **85.7% precision at <1 km** on accepted panoramas (typically tens of meters)
- **100% accuracy** on synthetic queries up to 1.0° noise
- **Zero impact** from uint8 quantization storage format

## Quick Start

```bash
# verify algorithm correctness (11/11 tests):
conda run -n skyline_env python tests/test_core.py

# full methodology and results:
cat ARCHITECTURE.md

# run evaluation:
python archive/scripts/gsv_improve_eval.py --stride 2
python archive/scripts/offgrid_synthetic_eval.py --samples 20 --stride 20
```
