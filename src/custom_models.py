"""
Custom estimators that must be importable at unpickle time.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor


class KNNBaggingRegressor(BaseEstimator, RegressorMixin):
    """
    Bagged KNN baseline: several KNN models trained on bootstrap resamples of
    the rows. Included because KNN groups by feature-space proximity, which
    mimics topological awareness without any explicit graph features.
    """
    def __init__(self, n_estimators: int = 15, n_neighbors: int = 40, seed: int = 42):
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        self.n_features_in_ = X.shape[1]
        for _ in range(self.n_estimators):
            idx = rng.integers(0, len(y), len(y))
            m = KNeighborsRegressor(
                n_neighbors=self.n_neighbors, weights="distance", n_jobs=2)
            m.fit(X[idx], y[idx])
            self.models_.append(m)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.column_stack([m.predict(X) for m in self.models_])
        return preds.mean(axis=1)
