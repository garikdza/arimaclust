import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils import Tags
from sklearn.utils.validation import validate_data

from arimaclust.metrics import ARMADistance


class DistanceInputMixin:
    """
    Resolve a clusterer's ``distance`` parameter into a distance matrix.

    ``distance`` is ``None`` (a default ``ARMADistance``), an ``ARMADistance`` to fit on the
    raw series, or ``'precomputed'`` when ``X`` already is a square distance matrix.
    """

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.distance == 'precomputed'
        return tags

    def _distance_matrix(self, X: ArrayLike) -> np.ndarray:
        """
        Fit a copy of ``distance`` on ``X`` (or validate a precomputed matrix) and set ``distance_``.

        Parameters
        ----------
        X : ArrayLike
            raw series, or a square distance matrix when ``distance='precomputed'``

        Returns
        -------
        dist_matrix : np.ndarray
            square distance matrix
        """
        if self.distance == 'precomputed':
            D = validate_data(self, X)
            if D.shape[0] != D.shape[1]:
                raise ValueError(f"distance='precomputed' needs a square matrix, got shape {D.shape}")
            self.distance_ = None
            return D
        if not (self.distance is None or isinstance(self.distance, ARMADistance)):
            raise ValueError(f"distance must be None, 'precomputed' or an ARMADistance, got {self.distance!r}")

        distance = ARMADistance() if self.distance is None else clone(self.distance)
        self.distance_ = distance.fit(X)
        return np.asarray(distance.dist_matrix_)
