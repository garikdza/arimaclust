# arimaclust

[![codecov](https://codecov.io/gh/garikdza/arimaclust/branch/main/graph/badge.svg)](https://codecov.io/gh/garikdza/arimaclust)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

arimaclust is a Python library for model-based time series clustering.

<p align="center">
  <img src="examples/synthetic/figures/accuracy_vs_length.png" width="640"
       alt="AMI against series length for four synthetic three-cluster sets, rising to 1 sooner for larger separation">
</p>
<p align="center"><sub>AMI against series length on four synthetic three-cluster sets of separation δ, the smallest
Piccolo distance between the generating processes.</sub></p>

## Features

**Distances** between ARIMA models:

| `metric`              | distance                                                               | reference                                                            |
| --------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `'piccolo'` (default) | Euclidean distance between the AR(∞) coefficients                      | [Piccolo (1990)](https://doi.org/10.1111/j.1467-9892.1990.tb00048.x) |
| `'cepstral'`          | Euclidean distance between the cepstral coefficients of the AR(∞) form | [Kalpakis et al. (2001)](https://doi.org/10.1109/ICDM.2001.989529)   |
| `'mahalanobis'`       | Mahalanobis distance between the AR(∞) coefficients                    | [Maharaj (1996)](https://doi.org/10.1080/00949659608811737)          |

**Clustering** with scikit-learn compatible estimators on the distance matrix:

- `ConsistentClustering(n_clusters)`: farthest-point initialisation and nearest-centre assignment
  ([Khaleghi et al., 2016](https://www.jmlr.org/papers/v17/khaleghi16a.html)). With the Piccolo distance and BIC
  order selection it is weakly asymptotically consistent for ARMA processes
  ([Adamyan, 2023](https://doi.org/10.3103/S1068362323030020)).
- `KMedoids(n_clusters)`: k-medoids, starting from PAM BUILD (`init='build'`) or from the farthest-point centres
  (`init='farthest'`), then PAM swaps until no exchange of a medoid lowers the total distance to the medoids
  ([Kaufman & Rousseeuw, 1990](https://doi.org/10.1002/9780470316801)).
- `ThresholdClustering(alpha)`: connected components of the graph linking pairs at distance ≤ `alpha`, for an
  unknown number of clusters.

## Install

```bash
pip install git+https://github.com/garikdza/arimaclust
```

## Quickstart

```python
from arimaclust import ARMADistance, KMedoids
from arimaclust.data import generate_clusters

# one (AR, MA) coefficient pair per cluster: an AR(1), an AR(2) and an MA(1)
params = [([-0.6], []), ([-0.2, 0.7], []), ([], [0.2])]
# generate dataset
X, y = generate_clusters(params, length=500, n_per_cluster=20, random_state=0)
# init distance estimator
distance = ARMADistance(max_p=3, max_q=3, d=0, metric='piccolo', tau=50, n_jobs=-1)
```

From raw series, the clusterer fits the distance itself:

```python
model = KMedoids(n_clusters=3, distance=distance).fit(X)
model.labels_                  # cluster per series
model.distance_.models_        # fitted ARIMA per series
model.distance_.dist_matrix_   # pairwise Piccolo distances
```

From a precomputed distance matrix, for example to try several clusterers on one set of fits:

```python
D = distance.fit_transform(X)  # (60, 60) Piccolo distances
labels = KMedoids(n_clusters=3, distance='precomputed').fit_predict(D)
```

## Examples

[`examples/fred_md/`](examples/fred_md) clusters 121 monthly US macroeconomic series from FRED-MD
([McCracken & Ng, 2016](https://doi.org/10.1080/07350015.2015.1086655)) into four clusters, with the cepstral distance and `KMedoids`.

<p align="center">
  <img src="examples/fred_md/figures/distance_matrix.png" width="900"
       alt="Cepstral distance matrix of the FRED-MD series ordered by cluster beside a 2-D MDS embedding coloured by cluster">
</p>
<p align="center"><sub>Left: the cepstral distance matrix of the FRED-MD series, its rows and columns ordered by
cluster (lighter is closer). Right: a 2-D multidimensional scaling of the same distances, coloured by cluster, with the
four medoids circled and named.</sub></p>

[`examples/synthetic/`](examples/synthetic) evaluates the method on simulated sets with a controlled separation δ:
how accuracy grows with series length and δ, and how well the distance is estimated.

```bash
uv sync --extra examples
uv run --extra examples python examples/fred_md/clusters.py               # downloads the data the first time
uv run --extra examples python examples/synthetic/accuracy_vs_length.py   # also accuracy_vs_delta.py, distance_estimation.py, ...
```

## Citation

G. Adamyan, "Weakly consistent offline clustering of ARMA processes", *Journal of Contemporary Mathematical
Analysis*, 58(3), 183–190, 2023. https://doi.org/10.3103/S1068362323030020

## References

- G. Adamyan, "Model-based time series clustering", PhD thesis, Yerevan State University, 2023.
  https://www.ysu.am/en/councils/dissertations/52425
- D. Piccolo, "A distance measure for classifying ARIMA models", *Journal of Time Series Analysis*, 11(2),
  153–164, 1990. https://doi.org/10.1111/j.1467-9892.1990.tb00048.x
- K. Kalpakis, D. Gada, V. Puttagunta, "Distance measures for effective clustering of ARIMA time-series",
  *Proceedings of the 2001 IEEE International Conference on Data Mining*, 273–280, 2001.
  https://doi.org/10.1109/ICDM.2001.989529
- E. A. Maharaj, "A significance test for classifying ARMA models", *Journal of Statistical Computation and
  Simulation*, 54(4), 305–331, 1996. https://doi.org/10.1080/00949659608811737
- A. Khaleghi, D. Ryabko, J. Mary, P. Preux, "Consistent algorithms for clustering time series", *Journal of
  Machine Learning Research*, 17(3), 1–32, 2016. https://www.jmlr.org/papers/v17/khaleghi16a.html
- M. Corduas, D. Piccolo, "Time series clustering and classification by the autoregressive metric",
  *Computational Statistics & Data Analysis*, 52(4), 1860–1872, 2008. https://doi.org/10.1016/j.csda.2007.06.001
- J.-M. Bardet, K. Kamila, W. Kengne, "Consistent model selection criteria and goodness-of-fit test for common
  time series models", *Electronic Journal of Statistics*, 14(1), 2009–2052, 2020. https://doi.org/10.1214/20-EJS1709
- M. W. McCracken, S. Ng, "FRED-MD: A monthly database for macroeconomic research", *Journal of Business & Economic
  Statistics*, 34(4), 574–589, 2016. https://doi.org/10.1080/07350015.2015.1086655

## License

Released under the [MIT License](LICENSE), © 2023 [garikdza](https://github.com/garikdza).
