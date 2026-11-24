# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Consistent model-based clustering of ARIMA time series. Each series gets an auto-fitted ARMA model, the models are compared through their AR(∞) representations (the Piccolo distance by default, or the LPC cepstral or Maharaj Mahalanobis distance), and the series are clustered on the resulting distance matrix. The theory (assumptions and the consistency result) is in the paper cited in the README.

## Commands

The project is managed with uv (`pyproject.toml`, hatchling backend).

- `uv sync`: install the library plus the dev tools (pytest, pytest-cov, ruff).
- `uv run pytest`: run the tests, including sklearn's estimator checks (`tests/test_sklearn_checks.py`), but not the tests marked `slow`. For a single test use `uv run pytest tests/test_metrics.py::test_mahalanobis_tau1_hand_value`.
- `uv run pytest -m slow`: run the slow statistical tests (clustering recovery rising with series length). Add a node id after `-m slow` to run one; without `-m slow`, a slow test is deselected even when named by node id.
- `uv run pytest --cov`: run the tests with a coverage report for `arimaclust`. It fails below the 85% floor set in `pyproject.toml`, the same check CI runs. Add `--cov-report=html` for a browsable report.
- `uv run ruff check .`: lint (rules E, F, I, UP and B, line length 120, notebooks excluded). `src = [".", "examples"]` makes `synthetic`, `results` and `style` first-party imports.
- `uv sync --extra examples`: also install matplotlib, which only the example scripts need.
- `uv run --extra examples python examples/<script>.py`: run one example script. They need no network and no GPU, and write their figures to `examples/figures/`; only `accuracy_vs_length.png` and `accuracy_vs_delta.png` (shown in the README) are tracked, and `.gitignore` keeps the rest out. The four scripts that fit models save their raw results as `examples/figures/<script>.npz` and redraw from that file on later runs; add `--recompute` to fit again, as needed after changing a script's settings or a curated set (a changed setting is not detected).
- CI (`.github/workflows/ci.yml`) runs ruff and the default tests with coverage on pushes to `main` and on PRs, and uploads the report to Codecov.
- Nothing here needs a GPU. Prefix Python runs with `CUDA_VISIBLE_DEVICES=""`, as the global GPU rule requires.

## Tests

Tests are unit tests that mirror the code layout: one test file per module, at the mirrored path, named `test_<module>.py`, for example `examples/synthetic.py` → `tests/examples/test_synthetic.py`.

- `arimaclust`'s tests sit directly in `tests/`, one file per area: `test_metrics.py` for `arimaclust/metrics/`, `test_estimators.py` for the estimators, and so on.
- The `examples/` figure scripts are entry points and are run by hand. The modules they import, `synthetic.py`, `results.py` and `style.py`, are unit-tested.
- pytest's `pythonpath = ["examples"]` lets the tests `import synthetic`, `import results` and `import style`, as the scripts do. None of these modules may import matplotlib, so their tests run under a plain `uv sync`, as in CI.

## Architecture

The pipeline is `estimate → distance matrix → cluster`. The public API is the three estimators exported from `arimaclust/__init__.py`; everything else is internal.

- `distance.py`: `ARMADistance(max_p=3, max_q=3, d=0, metric='piccolo', tau=50, arima_kwargs=None, n_jobs=None)`, an sklearn transformer with `fit` and `fit_transform` only. It reads a list of 1-D series (lengths may differ), a 2D array (one series per row) or a DataFrame (one series per column, NaNs dropped), and validates every series before fitting any of them. It then fits the models, checks that each one is invertible (the error names the series), and fills `processes_` and `dist_matrix_` (a list and an ndarray; for a DataFrame, a `{column: process}` dict and a DataFrame labelled by the columns).
  - `arima_kwargs` go to `auto_arima` and override the fit defaults, but must not set `max_p`, `max_q` or `d`. `start_p` / `start_q` are clamped to the maximum orders.
  - `metric` picks `'piccolo'`, `'cepstral'` or `'mahalanobis'` and is checked before fitting. `'mahalanobis'` needs an integer `d` (`d=None` lets auto_arima pick it per series). `n_jobs` covers both the model fits and the distance matrix.
- `autoarima/estimation.py`: `estimate_processes(X, n_workers=8, **kwargs)` runs `pmdarima.auto_arima` on each series of a list in parallel (joblib) and returns a list of statsmodels `ArmaProcess`. It selects orders by **BIC by default**, which the consistency theory requires; AIC is not consistent.
  - `kwargs` go straight to `auto_arima` (`max_p`, `max_q`, `d`, ...) and override `fit_arma`'s fixed options. The differencing order `d` only affects fitting: the returned `ArmaProcess` keeps just the AR and MA parts.
- `metrics/`: distances between fitted models, all built on the AR(∞) coefficients and needing invertible processes. `tau` defaults to 50 everywhere; it is an integer ≥ 1, checked by every metric (through `check_tau` in `autoregressive.py`) and by `ARMADistance` before fitting.
  - `piccolo` compares the first `tau` AR(∞) coefficients π₁…π_tau, and `cepstral` the first `tau` LPC cepstral coefficients; both take two `ArmaProcess` objects.
  - `mahalanobis` is Maharaj's Wald statistic on the first `tau` AR(∞) coefficients, weighted by each series' autocovariances and length. It compares `SeriesModel(series, process)` items, which `ARMADistance` builds from the series differenced `d` times; each needs more than `tau` observations and must not be constant.
  - `matrix.dist_matrix_parallel` looks the metric up **by name** in the `arimaclust.metrics` namespace, so a new metric has to be exported in `metrics/__init__.py` and listed in `distance.METRICS`, the names `ARMADistance` accepts.
  - It computes the pairs with joblib (`n_jobs`, `None` runs serially). A dict input gives a DataFrame labelled by its keys (index and columns). A list input gives an ndarray.
- `models/`: sklearn clusterers taking `distance=None | ARMADistance(...) | 'precomputed'`. `models/distance_input.py`'s `DistanceInputMixin` fits a clone of `distance` (`None` means `ARMADistance()`) into `distance_`, or validates `X` as a square matrix for `'precomputed'` and sets the `pairwise` sklearn tag. The `distance` parameter itself is never fitted. Both clusterers fill `labels_` and provide `fit_predict`, but no `predict`.
  - `centroid.ConsistentClustering(n_clusters=2)` starts farthest-point selection at sample 0, then assigns each sample to its nearest centre (`centers_` holds the centre sample indices). This is the algorithm with the consistency proof.
  - `neighbours.ThresholdClustering(alpha)` builds the graph of pairs with distance ≤ `alpha` and returns its connected components as clusters. `alpha` has no default.
- `data/`: synthetic ARMA cluster generators, with parameter sets in `examples.py` stored as `(ar_params, ma_params)` tuples.

`examples/` (not in the wheel, no `__init__.py`) holds synthetic examples at a controlled separation δ, the smallest Piccolo distance (`tau=50`) between the processes that generate the clusters.

- `synthetic.py`: `BANDS` (four δ bands) and `band_of`, `CURATED_SETS` (one three-cluster `CuratedSet` per band, its δ recorded and checked in the tests), `min_separation`, `generate_separated_params(n_clusters, delta, seed)` (random stationary, invertible ARMA(≤2, ≤2) processes kept greedily while pairwise ≥ δ; raises `RuntimeError` after `max_draws`) and `clustering_scores` (one seeded draw, `ConsistentClustering` + default `ARMADistance` fit and AMI per seed). No matplotlib.
- `results.py`: `save_results` / `load_results` (a dict of arrays as a `.npz` file, no pickle, so object arrays are rejected at save), `load_or_compute(path, compute, recompute)` (computes and saves when the file is missing or `recompute` is set, then always returns what it reads back from the file) and `parse_args`, the `--recompute` flag. No matplotlib.
- `style.py`: the minimal figure style every script draws in, `MINIMAL` (matplotlib rc settings for `plt.rc_context`), and the colours: `BAND_COLOURS` (one blue per δ band, darker for larger δ), `CLUSTER_COLOURS` and `DISTANCE_RAMP`. No matplotlib.
- The scripts, each writing to `examples/figures/` in the `style.MINIMAL` style: `accuracy_vs_length.py` (AMI against length per band, the README's top figure), `accuracy_vs_delta.py` (AMI against δ on generated two-cluster sets, the README's second figure), `distance_matrix.py` (`distance_matrix.png`: heatmap and MDS embedding of a fitted distance matrix, misclustered series circled), `distance_estimation.py` (fitted against true δ by length) and `processes.py` (sample series and true AR(∞) coefficients). They import `synthetic`, `results` and `style` as sibling modules. The four that fit models (all but `processes.py`) have a `compute()` that fits and returns the raw arrays (every draw's AMI or every distance estimate, the full distance matrix, plus the labels, lengths and δs the plot needs), a `plot(results)` that draws only from those loaded arrays, and a `main()` that joins them through `results.load_or_compute`, saving to `figures/<script>.npz`. `processes.py` fits nothing and draws directly.

Exchange-rate data (apilayer) must never be committed; `.gitignore` blocks data and result files under `experiments/`.
