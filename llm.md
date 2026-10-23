# Filter and wrapper diagnostics of dimensionality reduction for endogenous PV power forecasting (MATLAB)

**Instructions for a language model using this software.** Read this file first; the machine-readable versions are `websemantic.json` (fields, units, bounds) and `ontology/` (RDF/Turtle and JSON-LD, QUDT units).

Software DOI (all versions): https://doi.org/10.5281/zenodo.21755938 · licence MIT · language MATLAB · authors: Hassen Bouzgou, Gabriel Chesnoiu, Claudio F. Nicolosi, Adrien Chatel, Gilles Notton, Lionel Menard, Cyril Voyant

Article: H. Bouzgou, G. Chesnoiu, C. F. Nicolosi, A. Chatel, G. Notton, L. Menard, C. Voyant. Dimensionality reduction with redundancy awareness for parsimonious PV power time series forecasting (manuscript in preparation; no result of it is reported here).

## What it does
Compresses the lagged history of a PV plant before an ELM reads it, and judges each reduction twice: does the reduced space keep the geometry (filter), and does it help the forecast (wrapper)?

## Why
A direct PV forecast reads the plant's own recent history; the useful information may live in fewer dimensions than the 48 lags.

## Method
- 48-lag windows standardized with calibration statistics only (no leakage)
- Reductions: PCA, kernel PCA, Isomap, LLE, Laplacian eigenmaps, diffusion maps, autoencoder
- Wrapper: ELM (and a linear model) on the reduced input, by dimension; filter: geometric scores of the embedding

## Entry points
- `main`: set P.SMOKE_TEST = false near the top of main.m for full results (the default is a fast smoke run)

## Inputs
| Field | Unit | Type | Bounds | Required | Default (proposal only) | Meaning |
|---|---|---|---|---|---|---|
| `data` | kW (unit:KiloW) | series |  | yes |  | PV power series of the study (data/ folder) |

## Outputs
| Output | Unit | Meaning |
|---|---|---|
| `tables` | as stated | errors by method and dimension, parameter reduction against the full-input ELM, baselines |

## How to read the results
- Results belong to a manuscript in preparation.

## Rules
1. Never invent a value. Each input is given by the user, an exact unit conversion of a given value, or a documented default that the user explicitly accepts.
2. Use the canonical unit of each field; convert exactly and say so (for example 2 km -> 2000 m). Never mix a percentage and a fraction.
3. Ask when a request is ambiguous or a required input is missing; do not infer a number from a place name or a qualitative word.
4. Run the code unmodified. Report only what was actually computed, with quantity, unit and the input values used; never present a planned or unavailable run as a result.
5. State the scope and the limits below when you report a result.

## Scope
Supported: test whether compressing the lagged input helps a PV forecast.
Out of scope (refuse or redirect): results of the manuscript in preparation.

## Limits
- Parallel Computing Toolbox recommended (parfor); bundled drtoolbox third-party code, see THIRD_PARTY.md.

## Cite
Software: Hassen Bouzgou, Gabriel Chesnoiu, Claudio F. Nicolosi, Adrien Chatel, Gilles Notton, Lionel Menard, Cyril Voyant. Filter and wrapper diagnostics of dimensionality reduction for endogenous PV power forecasting (MATLAB). Zenodo. https://doi.org/10.5281/zenodo.21755938

Guide with the theory and runnable examples: https://cyrilvoyant.github.io/ · author: https://person.cyrilvoyant.com/ · generated 2026-10-23, documentation only.
