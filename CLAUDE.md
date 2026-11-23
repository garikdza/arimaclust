# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Consistent model-based clustering of ARIMA time series. Each series gets an auto-fitted ARMA model, the models are compared through their AR(∞) representations (the Piccolo distance), and the series are clustered on the resulting distance matrix. The theory (assumptions and the consistency result) is in the paper cited in the README.

## Commands

The project is managed with uv (`pyproject.toml`, hatchling backend).

- `uv sync`: install the library plus the dev tools (pytest, ruff).
- `uv run pytest`: run the tests, including sklearn's estimator checks (`tests/test_sklearn_checks.py`). For a single test use `uv run pytest tests/test_smoke.py::test_piccolo_matches_known_value`.
- `uv run ruff check .`: lint (rules E, F, I, UP and B, line length 120, notebooks excluded).
- Nothing here needs a GPU. Prefix Python runs with `CUDA_VISIBLE_DEVICES=""`, as the global GPU rule requires.

## Architecture

The pipeline is `estimate → distance matrix → cluster`. The public API is the three estimators exported from `arimaclust/__init__.py`; everything else is internal.

- `distance.py`: `ARMADistance(max_p=3, max_q=3, d=0, tau=10, arima_kwargs=None, n_jobs=None)`, an sklearn transformer with `fit` and `fit_transform` only. It reads a list of 1-D series (lengths may differ), a 2D array (one series per row) or a DataFrame (one series per column, NaNs dropped), and validates every series before fitting any of them. It then fits the models, checks that each one is invertible (the error names the series), and fills `processes_` and `dist_matrix_` (a list and an ndarray; for a DataFrame, a `{column: process}` dict and a DataFrame labelled by the columns).
  - `arima_kwargs` go to `auto_arima` and override the fit defaults, but must not set `max_p`, `max_q` or `d`. `start_p` / `start_q` are clamped to the maximum orders.
  - `n_jobs` covers both the model fits and the distance matrix.
- `autoarima/estimation.py`: `estimate_processes(X, n_workers=8, **kwargs)` runs `pmdarima.auto_arima` on each series of a list in parallel (joblib) and returns a list of statsmodels `ArmaProcess`. It selects orders by **BIC by default**, which the consistency theory requires; AIC is not consistent.
  - `kwargs` go straight to `auto_arima` (`max_p`, `max_q`, `d`, ...) and override `fit_arma`'s fixed options. The differencing order `d` only affects fitting: the returned `ArmaProcess` keeps just the AR and MA parts.
- `metrics/`: distances between fitted models, all built on the AR(∞) coefficients and needing invertible processes. `tau` defaults to 50 everywhere.
  - `piccolo` compares the AR(∞) coefficients truncated at `tau` lags, and `cepstral` the first `tau` LPC cepstral coefficients; both take two `ArmaProcess` objects.
  - `mahalanobis` is Maharaj's Wald statistic on the first `tau` AR(∞) coefficients, weighted by each series' autocovariances and length. It compares `SeriesModel(series, process)` items; each series needs more than `tau` observations and must not be constant.
  - `matrix.dist_matrix_parallel` looks the metric up **by name** in the `arimaclust.metrics` namespace, so a new metric has to be exported in `metrics/__init__.py`.
  - It computes the pairs with joblib (`n_jobs`, `None` runs serially). A dict input gives a DataFrame labelled by its keys (index and columns). A list input gives an ndarray.
- `models/`: sklearn clusterers taking `distance=None | ARMADistance(...) | 'precomputed'`. `models/distance_input.py`'s `DistanceInputMixin` fits a clone of `distance` (`None` means `ARMADistance()`) into `distance_`, or validates `X` as a square matrix for `'precomputed'` and sets the `pairwise` sklearn tag. The `distance` parameter itself is never fitted. Both clusterers fill `labels_` and provide `fit_predict`, but no `predict`.
  - `centroid.ConsistentClustering(n_clusters=2)` starts farthest-point selection at sample 0, then assigns each sample to its nearest centre (`centers_` holds the centre sample indices). This is the algorithm with the consistency proof.
  - `neighbours.ThresholdClustering(alpha)` builds the graph of pairs with distance ≤ `alpha` and returns its connected components as clusters. `alpha` has no default.
- `data/`: synthetic ARMA cluster generators, with parameter sets in `examples.py` stored as `(ar_params, ma_params)` tuples.

Exchange-rate data (apilayer) must never be committed; `.gitignore` blocks data and result files under `experiments/`.
