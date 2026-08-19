# pv-dimensionality-reduction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21755939.svg)](https://doi.org/10.5281/zenodo.21755939)
**How much of the lagged photovoltaic (PV) signal do you actually need to forecast it?**

This repository contains the MATLAB code behind a study on short-term PV power
forecasting (30 min to 10 h ahead) that uses *only the plant's own past
production* — no irradiance, no numerical weather prediction, no exogenous
meteorology. The forecasting problem is treated as a geometric compression
problem: a one-day window of past power values is redundant, and we try to find
the small number of directions that really carry the forecastable information.

The point is not to squeeze out one more decimal of accuracy. It is to make the
reduction **legible** and to keep the loss of information under control, so that
a compact model can be defended against a black-box one on an explicit
**accuracy vs. footprint** trade-off.

---

## The idea in one paragraph

The candidate predictor space is a sliding window of `LB = 48` past power values
sampled every 30 minutes (one diurnal cycle). We reduce that space with seven
classical techniques — PCA, Kernel PCA, Isomap, LLE, Laplacian Eigenmaps,
Diffusion Maps and an Autoencoder — and we judge each reduced representation
from two independent angles:

- a **filter** diagnostic, which measures how well the reduced space preserves
  the *geometry* of the original lagged cloud (local neighbourhoods +
  global distances), with no forecasting model involved;
- a **wrapper** diagnostic, which measures how *useful* the reduced space is for
  *forecasting*, by coupling it to an Extreme Learning Machine (ELM).

Keeping the two apart is deliberate: a representation can preserve geometry
without being ideal for prediction, and vice versa. Both are reported against
the same baselines and, crucially, against the number of parameters each model
uses — so the effect of dimensionality reduction is read directly off the table.

---

## Quick start

You need MATLAB (developed and tested on R2025b) with the **Parallel Computing
Toolbox** (the loops over dimensions run in `parfor`; without the toolbox they
fall back to serial execution).

```matlab
% From MATLAB, with this folder as the current directory:
main
```

That's it. `main.m` locates itself, puts the bundled `drtoolbox/` on the path,
finds the dataset under `data/`, and writes two result tables. **Nothing to
configure** — no toolbox path, no data path.

The first run defaults to a fast *smoke* configuration (`P.SMOKE_TEST = true`)
that finishes in a couple of minutes and only checks that everything runs. For
real results, set `P.SMOKE_TEST = false` near the top of `main.m` (this uses two
years of data and five horizons, and is meant for a workstation or a cluster —
it takes a while).

---

## What you get

Two CSV/MAT tables, one per diagnostic, sharing the exact same baseline rows:

- `Results_wrapper.csv` — for every method, the reduced dimension that minimises
  the forecasting error (`NICE_Sigma`), plus a linear counterpart (`*_AR`);
- `Results_filter.csv` — for every method, the dimension selected on the
  geometric score alone (and, for comparison, the forecast that this
  geometrically-chosen dimension produces with the same ELM).

Both tables are sorted by `N_params`, so you can literally scan down the column
and watch the error move as the model grows or shrinks. Each row carries:

| Column | Meaning |
|---|---|
| `Method` | baseline, DR method (`PCA`, …), or its linear variant (`PCA_AR`) |
| `BestDim` | selected reduced dimension `d` |
| `N_params` | number of trainable parameters of the model |
| `Param_reduction_pct` | footprint reduction vs. the full-input ELM |
| `FilterScore` | geometric score `S(d*)` (filter table only) |
| `nRMSE`, `R2` | standard error metrics |
| `NICE1/2/3`, `NICE_Sigma` | informed error vs. persistence (see below) |

### Baselines (identical in both tables)

| Model | Definition | Footprint |
|---|---|---|
| `Persistence_P` | `x̂(t+h) = x(t)` | 0 |
| `Persistence_Pcyclic` | `x̂(t+h) = x(t+h−T)`, `T` = 1 day | 0 |
| `BLEND_tilde` | phase-weighted blend of the two persistences | phase weights |
| `AR_full` | linear least squares on the 48 lags (+ time features) | `D+1` |
| `ELM_full` | ELM on the 48 lags (+ time features), no reduction | `Nh·(D+2)` |

By construction the simple persistence has `NICE^k = 1` at every order — it is
the denominator of the NICE metrics, so a model is "good" when its NICE is below 1.

Since there is nothing to forecast at night, the metrics are computed on
daytime samples only (solar elevation > 0 at the target time). `P.night`
selects the behaviour: `'day'` (default) evaluates daytime only, `'zero'`
clamps the night forecasts to zero and keeps every sample, `'all'` keeps
everything untouched. Set the site coordinates with `P.lat` / `P.lon`, and make
sure the timestamps are in **UTC** (convert local-time datasets first, e.g.
Alice Springs).

---

## How it works

The pipeline is deliberately split into small, reusable functions that `main.m`
orchestrates. For each horizon:

1. **`prepare_supervised.m`** builds the supervised problem (48-lag windows →
   target), splits it chronologically into calibration/evaluation, and
   standardises the inputs using **calibration statistics only** (no leakage
   through the normalisation). Optional deterministic time features — hour of
   day and day of year, encoded as sin/cos on the *target* timestamp — are
   prepared here.
2. **`compute_references.m`** evaluates the five baselines once (shared by both
   diagnostics).
3. **`dr_embed.m`** computes each method's embedding once. Nonlinear methods are
   fitted on a subsample of *landmarks* drawn from the calibration set (which
   keeps the eigen-decomposition affordable) and then extended to every point,
   either with the toolbox's native out-of-sample operator or, when that is not
   available or not numerically safe, with a k-NN (Nyström-like) interpolation.
   The same embedding is handed to both the filter and the wrapper, so the two
   analyses see exactly the same reduced space.
4. **`DR_wrapper.m`** sweeps the reduced dimension, trains an ELM (and a linear
   AR) on the reduced space augmented with the time features, and keeps the
   dimension that minimises `NICE_Sigma`.
5. **`DR_filter.m`** sweeps the same dimensions and scores each one geometrically
   with `S(d) = α·T(k) + (1−α)·ρ` (trustworthiness + a Mantel-type distance
   correlation). Because `S(d)` grows mechanically towards the full dimension, we
   do **not** take its maximum; we keep the *smallest* dimension that reaches a
   fraction `q` of the achievable score — a simple, transparent parsimony rule.
   The full `S(d)` curve is saved as well, in case a Pareto plot is wanted later.
   The forecast at that single geometrically-chosen dimension is then evaluated
   with the same ELM (`P.filter_eval`), so the filter table can be read next to
   the wrapper table on the same error scale.

Two notes worth keeping in mind:

- The dimension is selected on the same evaluation set that is reported (a
  two-way split, not a three-way one). This is a conscious choice; the wrapper
  side therefore carries a mild optimistic selection bias, whereas the filter
  side, being model-free, does not.
- The reduction only ever compresses the 48 lags. The time features are appended
  *after* reduction and never fed into the DR step.

### Reduction techniques

| Method | Nature | Out-of-sample extension |
|---|---|---|
| PCA | linear, spectral | native (exact projection) |
| Kernel PCA | nonlinear (kernel) | native, fitted on landmarks |
| Isomap | geodesic manifold | k-NN interpolation from landmarks |
| LLE | local linear patches | k-NN interpolation from landmarks |
| Laplacian Eigenmaps | graph spectral | k-NN interpolation from landmarks |
| Diffusion Maps | Markov diffusion | k-NN interpolation from landmarks |
| Autoencoder | nonlinear encoder–decoder | native, one training per dimension |

The manifold methods (Isomap, LLE, Laplacian, Diffusion Maps) are extended by a
distance-weighted k-NN interpolation from the landmarks rather than by the
toolbox's native out-of-sample operator: the latter loops point by point
(Dijkstra, local inversions), which is prohibitive on ~10⁴ evaluation points and
numerically unstable for LLE. PCA and Kernel PCA keep their exact, vectorised
projections; the autoencoder uses its trained decoder.

---

## Parameters

Everything is set in the `P` struct at the top of `main.m`. The most useful:

| Field | What it controls |
|---|---|
| `SMOKE_TEST` | fast sanity run vs. full run |
| `FH_list` | forecast horizons, in 30-min steps (`[1 2 6 12 20]` = 0.5–10 h) |
| `LB` | input window length (48 = one day) |
| `ratio` | calibration fraction of the chronological split |
| `techniques` | which DR methods to run |
| `USE_TEMPORAL` | append hour/day features to the ELM/AR inputs |
| `ridge` | Tikhonov regularisation of the ELM output weights (0 = plain) |
| `N_ELM_hidden`, `N_ELM_candidates` | ELM width and number of random draws |
| `L_MAX` | number of landmarks for the nonlinear embeddings |
| `filter_nmax` | subsample size for the geometric score (it is O(n²)) |
| `filter_alpha`, `filter_k`, `filter_q` | filter score weighting, neighbourhood, and selection threshold |
| `run_wrapper`, `run_filter` | run either or both diagnostics |
| `parallel` | `true` uses a parallel pool; `false` runs everything serially |
| `nworkers` | requested pool size (capped to the machine's cores) |

## Long runs

The full configuration (two years, five horizons) is a multi-hour job, so the
script is built to survive interruptions:

- it prints where it is and how long each step took —
  `embed 12/35 | FH 3.0h | Laplacian | 8.2 min`, then per-phase and per-horizon
  timings;
- after **each horizon** it writes a checkpoint (`Results_checkpoint.mat`) and
  the partial CSVs, and on the next launch it **reloads and skips the horizons
  already done**, so a crash never sends you back to zero;
- the parallel pool is created with its idle timeout disabled, which avoids the
  classic "pool shut down" failure during long serial phases;
- the verbose toolbox output (welcome banners, autoencoder training traces,
  near-singular warnings) is muted through `run_quiet` — a small `evalc`
  wrapper — so the terminal shows only the progress lines above.

---

## Data

The `data/` folder ships with one site, **Palaiseau (France)**, resampled to a
30-minute grid. It is openly available and can be cited as:

> J. Badosa, C. Teissedre. *E4C Multivariable energy and meteorological dataset
> for a tertiary building* (2025). doi:10.14768/211DFF87-8187-447A-8086-1EB7C93A3688

The study behind this code also uses two further sites, which you can obtain
from their original providers and drop into `data/` (then point `P.fileName` at
them):

- **Risø (Denmark)** — SOLETE dataset, Pombo et al., doi:10.1016/j.dib.2022.108046
- **Alice Springs (Australia)** — Desert Knowledge Australia Solar Centre,
  http://dkasolarcentre.com.au/download

The quality-controlled, FAIR-formatted (NetCDF-CF) versions produced for the
paper are distributed through the *webservice-energy* infrastructure
(THREDDS / OGC catalogue): https://tds.webservice-energy.org/thredds/catalog/dimred/catalog.html

The CSV format expected by `data30min.m` is a two-column file
(`datetime, power`); keep that layout for any new site.

---

## Dependencies and credits

The nonlinear reductions rely on the **Matlab Toolbox for Dimensionality
Reduction (drtoolbox, v0.8b)** by **Laurens van der Maaten** (Delft University
of Technology), bundled here in `drtoolbox/` so the project runs out of the box:

> L. van der Maaten, *Matlab Toolbox for Dimensionality Reduction*, Delft
> University of Technology, 2007–2014. https://lvdmaaten.github.io/drtoolbox/
>
> L. van der Maaten, E. O. Postma, J. van den Herik, *Dimensionality Reduction:
> A Comparative Review*, 2008.

The toolbox keeps its **own, non-commercial license** — it is *not* covered by
the MIT license of this project. Please read `drtoolbox/NOTICE.txt` before any
non-academic use.

The forecasting and evaluation conventions build on:

> C. Voyant et al., *NICEk metrics: Unified and multidimensional framework for
> evaluating deterministic solar forecasting accuracy*, Sustainable Energy
> Technologies and Assessments 83 (2025) 104588. doi:10.1016/j.seta.2025.104588
>
> C. Voyant et al., *Symmetry-constrained forecasting of periodically correlated
> energy processes*, Applied Mathematical Modelling (2026). doi:10.1016/j.apm.2026.116988

---

## Citing this work

If this code is useful to you, please cite the software (see `CITATION.cff`,
which GitHub renders as a "Cite this repository" button) and the associated
article once it is available.

---

## Authors

Hassen Bouzgou (University of Batna 2, Algeria) · Gabriel Chesnoiu (Mines Paris,
PSL) · Claudio F. Nicolosi (University of Catania) · Adrien Chatel (University of
Corsica) · Gilles Notton (University of Corsica) · Lionel Menard (Mines Paris,
PSL) · Cyril Voyant (Mines Paris, PSL) — https://www.cyrilvoyant.com

## License

Project code: MIT (see `LICENSE`). Bundled drtoolbox: non-commercial, see
`drtoolbox/NOTICE.txt`.
