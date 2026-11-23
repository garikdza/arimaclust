from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import parametrize_with_checks

from arimaclust import ARMADistance, ConsistentClustering, ThresholdClustering

PRECOMPUTED_ONLY = ('check_clustering fits raw (50, 2) blob features regardless of the pairwise tag, '
                    'so the precomputed distance matrix is not square')


def _expected_failed_checks(estimator: BaseEstimator) -> dict[str, str]:
    if isinstance(estimator, ARMADistance):
        reason = 'a sample is a whole series and series may differ in length, so there is no fixed feature count'
        return {'check_n_features_in': reason, 'check_n_features_in_after_fitting': reason}
    return {'check_clustering': PRECOMPUTED_ONLY}


@parametrize_with_checks([ARMADistance(n_jobs=-1),
                          ConsistentClustering(distance='precomputed'),
                          ThresholdClustering(alpha=0.5, distance='precomputed')],
                         expected_failed_checks=_expected_failed_checks)
def test_sklearn_compatible_estimator(estimator: BaseEstimator, check: Callable[[BaseEstimator], None]):
    check(estimator)
