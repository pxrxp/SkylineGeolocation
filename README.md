# Skyline Geolocation — Khumbu (Everest Region)

Localize street-level photos from the **mountain skyline alone** — no GPS,
no landmarks, no roads or buildings. Region of study: Khumbu Himalaya,
Nepal (fog, steep relief, 30 m global DEM).

> **Deliverable:** everything needed for the report lives in
> [`final/`](final/) — start with [`final/README.md`](final/README.md)
> and [`final/METHODOLOGY.md`](final/METHODOLOGY.md).
>
> **Headline:** when three independent skyline matchers agree, the system
> pinpoints the camera to **<1 km at 86% precision (median ≈ 40 m)** and
> abstains otherwise.

## Repository layout

```
src/            production library (segmentation, matching, profiles, eval)
notebooks/      numbered study notebooks (region → database → data → training → matching → results)
scripts/        curated pipeline + evaluation scripts
scripts/archive/ one-off experiments & diagnostics (kept for reference)
tests/          core-algorithm logic tests (vs brute-force references)
final/          self-contained deliverable: methodology, results, figures
docs/           historical planning docs & experiment worklog
colab/          Colab-era working notebooks
app/            demo app
data/           datasets & DB (gitignored; see .gitignore)
```

## Setup

```bash
conda env create -f environment.yml
conda activate skyline_env
bash setup.sh          # HORAYZON build + data downloads if needed
```

## Typical workflow

1. `notebooks/01_RegionStudy/` — define region, download DEMs, viewshed
2. `notebooks/02_SkylineDatabase/` — generate the horizon database
   (1.34M viewpoints, Copernicus GLO-30 via HORAYZON)
3. `notebooks/03_SyntheticData/` + `04_SkySegmentation/` — training data
   and U-Net sky segmentation
4. `notebooks/05_SkylineMatching/` — matching evaluation & sensor study
5. `notebooks/06_GSV_Evaluation/` — final GSV benchmark & report figures

## Verification

```bash
python tests/test_core.py     # 11 logic tests vs brute-force references
```
