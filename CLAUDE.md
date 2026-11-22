# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Consistent model-based clustering of ARIMA time series. Each series gets an auto-fitted ARMA model, the models are compared through their AR(∞) representations (the Piccolo distance), and the series are clustered on the resulting distance matrix. The theory (assumptions and the consistency result) is in the paper cited in the README.

## Commands

The project is managed with uv (`pyproject.toml`, hatchling backend).

- `uv sync`: install the library plus the dev tools (pytest, ruff).
- `uv run pytest`: run the tests. For a single test use `uv run pytest tests/test_smoke.py::test_piccolo_matches_known_value`.
- `uv run ruff check .`: lint (rules E, F, I, UP and B, line length 120, notebooks excluded).
- Nothing here needs a GPU. Prefix Python runs with `CUDA_VISIBLE_DEVICES=""`, as the global GPU rule requires.

## Architecture

The pipeline is `estimate → distance matrix → cluster → (track)`:

- `autoarima/estimation.py`: `estimate_processes(X, n_workers=8, **process_args)` runs `pmdarima.auto_arima` on each series in parallel (joblib) and returns each fit as a statsmodels `ArmaProcess`. It selects orders by **BIC by default**, which the consistency theory requires; AIC is not consistent.
  - A list or 2D-array input gives a list of processes. A `DataFrame` input is handled column by column, with NaNs dropped, and gives a `{column: process}` dict.
  - `process_args` go straight to `auto_arima` (`max_p`, `max_q`, `d`, ...). The differencing order `d` only affects fitting: the returned `ArmaProcess` keeps just the AR and MA parts.
- `metrics/`: distances between two `ArmaProcess` objects. `piccolo` truncates the AR(∞) coefficients at `tau` lags and needs invertible processes.
  - `matrix.dist_matrix_parallel` looks the metric up **by name** in the `arimaclust.metrics` namespace, so a new metric has to be exported in `metrics/__init__.py`.
  - A dict input gives a DataFrame labelled by its keys (index and columns). A list input gives an ndarray.
- `models/base.py`: `ClusteringModel` holds the defaults (`tau=10`; `max_p=max_q=3`, `d=0`). It fills `processes_`, `dist_matrix` and `labels_`, and provides `fit_predict`. With `metric='precomputed'`, `X` is taken to be a ready distance matrix.
  - `centroid.ConsistentARMAClustering(k)` starts farthest-point selection at sample 0, then assigns each sample to its nearest centre. This is the algorithm with the consistency proof.
  - `neighbours.NearestNeighbourModel(alpha)` builds the graph of pairs with distance ≤ `alpha` and returns its connected components as clusters.
- `tracking/mapping.py`: `mapp_hungarian` matches clusters between two labelings, for example consecutive time windows.
- `data/`: synthetic ARMA cluster generators, with parameter sets in `examples.py` stored as `(ar_params, ma_params)` tuples.

Exchange-rate data (apilayer) must never be committed; `.gitignore` blocks data and result files under `experiments/`.
