# arimaclust

Consistent model-based clustering of ARIMA time series.

Each series gets an ARMA model, selected by BIC with `pmdarima.auto_arima`. Series are compared through the
[Piccolo (1990)](https://doi.org/10.1111/j.1467-9892.1990.tb00048.x) AR distance: the Euclidean distance between
the AR(∞) coefficients of the fitted models. They are then clustered with farthest-point initialisation and
nearest-centre assignment ([Khaleghi et al., JMLR 2016](https://www.jmlr.org/papers/v17/khaleghi16a.html)). When
the maximum orders and the number of clusters are known, this procedure is weakly asymptotically consistent
(Adamyan, *Journal of Contemporary Mathematical Analysis* 58(3), 2023).

## Install

```bash
uv sync                      # library + dev tools
uv sync --extra experiments  # + matplotlib, jupyter, etc. for experiments/
```

## Quickstart

```python
from arimaclust.data.examples import CLUSTER1
from arimaclust.data.generate import generate_clusters
from arimaclust.models.centroid import ConsistentARMAClustering

X, y = generate_clusters(params=CLUSTER1, t_min=500, samples_per_cluster=20)
model = ConsistentARMAClustering(k=len(CLUSTER1),
                                 process_args={'max_p': 3, 'max_q': 3, 'd': 0},
                                 metric_args={'tau': 10}).fit(X)
model.labels_       # cluster per series
model.dist_matrix   # pairwise Piccolo distances; usable with any precomputed-metric clusterer
```

## Experiments

- `experiments/synthetic/`: clustering synthetic ARMA clusters as the series length grows.
- `experiments/exchange/`: dynamic clustering of the foreign exchange market. Fetch the rates with
  `download_data.py`, which needs an apilayer `API_KEY` in a local `.env`.

## Citation

If you use arimaclust, please cite:

G. Adamyan, "Weakly consistent offline clustering of ARMA processes", *Journal of Contemporary Mathematical
Analysis*, 58(3), 183–190, 2023. https://doi.org/10.3103/S1068362323030020

## References

- G. Adamyan, "Model-based time series clustering", PhD thesis, Yerevan State University, 2023.
  https://www.ysu.am/en/councils/dissertations/52425
- G. Adamyan, "Model-based clustering of foreign exchange market", *Proceedings of the Engineering Academy of
  Armenia*, 20(1), 24–33, 2023.
- D. Piccolo, "A distance measure for classifying ARIMA models", *Journal of Time Series Analysis*, 11(2),
  153–164, 1990. https://doi.org/10.1111/j.1467-9892.1990.tb00048.x
- M. Corduas, D. Piccolo, "Time series clustering and classification by the autoregressive metric",
  *Computational Statistics & Data Analysis*, 52(4), 1860–1872, 2008. https://doi.org/10.1016/j.csda.2007.06.001
- A. Khaleghi, D. Ryabko, J. Mary, P. Preux, "Consistent algorithms for clustering time series", *Journal of
  Machine Learning Research*, 17(3), 1–32, 2016. https://www.jmlr.org/papers/v17/khaleghi16a.html
- J.-M. Bardet, K. Kamila, W. Kengne, "Consistent model selection criteria and goodness-of-fit test for common
  time series models", *Electronic Journal of Statistics*, 14(1), 2009–2052, 2020. https://doi.org/10.1214/20-EJS1709

## License

MIT
