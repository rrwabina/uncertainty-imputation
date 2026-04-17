"""Extending the library: custom models and acquisition functions.

Demonstrates how to:

1. Register a new acquisition function.
2. Register a new uncertainty-aware model.
3. Use them inside :class:`UncertaintyImputer` via their string aliases.

Run with::

    python examples/04_custom_components.py
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from uncertainty_imputation import (
    AcquisitionFunction,
    MCARAmputer,
    UncertaintyImputer,
    UncertaintyModel,
    register_acquisition,
    register_model,
)


# -----------------------------------------------------------------------
# 1. Custom acquisition: Upper Confidence Bound (for maximisation), here
#    adapted as "low confidence bound" for our minimisation convention.
# -----------------------------------------------------------------------

class LowerConfidenceBound(AcquisitionFunction):
    '''LCB: return ``-(mean - beta * std)``.

    Higher score = smaller mean and/or larger std. Used as an exploration-
    tilted score for minimisation targets.
    '''

    name = 'lcb'

    def __init__(self, beta: float = 2.0) -> None:
        self.beta = float(beta)

    def score(self, mean, std, y_best=None):
        return -(mean - self.beta * std)


# -----------------------------------------------------------------------
# 2. Custom model: bagged K-nearest-neighbours with bootstrap uncertainty.
# -----------------------------------------------------------------------

class BaggedKNNModel(UncertaintyModel):
    '''K-NN regressor with bootstrap-based uncertainty.'''

    name = 'bagged_knn'

    def __init__(
        self,
        n_bootstrap: int = 10,
        n_neighbors: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._models: list[KNeighborsRegressor] = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(self.random_state)
        self._models = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, X.shape[0], size=X.shape[0])
            # Adapt n_neighbors to the bootstrap sample size
            k = min(self.n_neighbors, len(np.unique(idx)))
            m = KNeighborsRegressor(n_neighbors=max(k, 1))
            m.fit(X[idx], y[idx])
            self._models.append(m)
        return self

    def predict(self, X):
        m, _ = self.predict_with_uncertainty(X)
        return m

    def predict_with_uncertainty(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.stack([m.predict(X) for m in self._models], axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1) if self.n_bootstrap > 1 else np.zeros_like(mean)
        return mean, std


def main() -> None:
    # Register the custom components under string aliases.
    register_acquisition('lcb', LowerConfidenceBound)
    register_model('bagged_knn', BaggedKNNModel)

    # Build a toy dataset with missingness.
    rng = np.random.default_rng(0)
    X_true = rng.normal(size=(300, 4))
    X_true[:, 1] = 0.7 * X_true[:, 0] + 0.4 * rng.normal(size=300)
    X_missing = MCARAmputer(missing_rate=0.2, random_state=0).ampute(X_true)

    # Use the custom components via their aliases.
    imputer = UncertaintyImputer(
        model='bagged_knn',
        acquisition='lcb',
        max_iter=3,
        random_state=42,
        model_kwargs={'n_bootstrap': 8, 'n_neighbors': 7},
        acquisition_kwargs={'beta': 2.0},
    )
    X_imputed = imputer.fit_transform(X_missing)
    print(f'Imputed dataset shape: {X_imputed.shape}')
    print(f'Iterations: {imputer.n_iter_}')
    print(f'Any remaining NaNs? {np.isnan(X_imputed).any()}')


if __name__ == '__main__':
    main()
