# Normalizing flows for geophysics — five lessons

Open the notebooks with the **"Python (normflows)"** kernel (the repo's `.venv`). Run them in order; later lessons reuse earlier results.

| Notebook | What it teaches | First run | Later runs |
|---|---|---|---|
| `01_flow_basics.ipynb` | Flow mechanics on a 2D toy: invertibility, `log_det`, densities vs. region probabilities | ~1 min | same |
| `02_buried_magnet_recipe_b.ipynb` | Recipe B (`reverse_kld`) on a synthetic buried magnet; graded against MCMC; lying about noise | ~5 min | same |
| `03_conditional_flow_recipe_a.ipynb` | Recipe A conditional flow trained on simulations; coverage test; amortization gap + fine-tuning; broken simulator | ~13 min | ~2 min (cached) |
| `04_osborne_real_data.ipynb` | Real airborne magnetics over the Osborne Cu-Au mine; block physics; 1- vs 2-block models; held-out lines; published geology | ~15 min | ~2 min (cached) |
| `05_drill_decision.ipynb` | Correlated-misfit diagnosis and tempering; hit-probability maps; expected value; value of information; drill-and-update | ~12 min | <1 min (cached) |

Shared code: `geomag.py` (sphere physics, MCMC, spline-flow builder) and `osborne.py` (data loading, verified prism physics, priors, warm-started training).
Trained flows are cached in `models/`; set `RETRAIN = True` in a notebook to retrain from scratch.

Extra packages used beyond the repo's requirements: `ensaio` (downloads the Osborne survey, CC-BY Geoscience Australia), `scipy`, `pandas`.

Data: Osborne airborne magnetic survey (Queensland Government, 1990) via [ensaio](https://www.fatiando.org/ensaio). Field values from IGRF (via `ppigrf`), hard-coded in `osborne.py`.
