"""Uncertainty-aware model wrappers.

Every model exposes three methods:

* :meth:`fit(X, y)`
* :meth:`predict(X)`               — point predictions
* :meth:`predict_with_uncertainty(X)` — ``(mean, std)`` tuple

The wrappers convert heterogeneous uncertainty signals (bootstrap residuals,
ensemble variance, leaf-level variance) into a common ``(mean, std)``
interface that acquisition functions can consume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - exercised only when xgboost missing
    _HAS_XGBOOST = False


class UncertaintyModel(ABC):
    '''Abstract base class for uncertainty-aware regressors.'''

    name: str = 'base'

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'UncertaintyModel':
        '''Fit the underlying estimator.'''

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Return point predictions.'''

    @abstractmethod
    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Return ``(mean, std)`` predictions.'''


# ---------------------------------------------------------------------------
# Linear regression with bootstrap residual uncertainty
# ---------------------------------------------------------------------------

class LinearRegressionModel(UncertaintyModel):
    '''Linear regression with bootstrap-based uncertainty.

    Uncertainty is estimated as the standard deviation across ``n_bootstrap``
    linear models fit on resampled training data.
    '''

    name = 'linear'

    def __init__(
        self,
        n_bootstrap: int = 25,
        random_state: Optional[int] = None,
        **estimator_kwargs,
    ) -> None:
        self.n_bootstrap = int(n_bootstrap)
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs
        self._models: list[LinearRegression] = []
        self._residual_std: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionModel':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        self._models = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            model = LinearRegression(**self.estimator_kwargs)
            model.fit(X[idx], y[idx])
            self._models.append(model)
        # Residual std on a single in-sample fit for a fallback when bootstrap
        # variance collapses to zero (e.g. perfectly collinear data).
        base = LinearRegression(**self.estimator_kwargs).fit(X, y)
        residuals = y - base.predict(X)
        self._residual_std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(X)
        return mean

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._models:
            raise RuntimeError('LinearRegressionModel must be fitted first')
        X = np.asarray(X, dtype=float)
        preds = np.stack([m.predict(X) for m in self._models], axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1) if self.n_bootstrap > 1 else np.zeros_like(mean)
        if np.allclose(std, 0.0) and self._residual_std > 0:
            std = np.full_like(mean, self._residual_std)
        return mean, std


# ---------------------------------------------------------------------------
# Decision tree with bootstrap uncertainty
# ---------------------------------------------------------------------------

class DecisionTreeModel(UncertaintyModel):
    '''Decision tree regressor with bootstrap-based uncertainty.

    A single decision tree is deterministic given its hyperparameters, so we
    train ``n_bootstrap`` trees on bootstrap samples to obtain a distribution
    over predictions, mirroring a small random forest but with the familiar
    "decision tree" semantics.
    '''

    name = 'decision_tree'

    def __init__(
        self,
        n_bootstrap: int = 25,
        random_state: Optional[int] = None,
        **estimator_kwargs,
    ) -> None:
        self.n_bootstrap = int(n_bootstrap)
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs
        self._models: list[DecisionTreeRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeModel':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31 - 1, size=self.n_bootstrap)
        n = X.shape[0]
        self._models = []
        for seed in seeds:
            idx = rng.integers(0, n, size=n)
            model = DecisionTreeRegressor(
                random_state=int(seed), **self.estimator_kwargs
            )
            model.fit(X[idx], y[idx])
            self._models.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(X)
        return mean

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._models:
            raise RuntimeError('DecisionTreeModel must be fitted first')
        X = np.asarray(X, dtype=float)
        preds = np.stack([m.predict(X) for m in self._models], axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1) if self.n_bootstrap > 1 else np.zeros_like(mean)
        return mean, std


# ---------------------------------------------------------------------------
# Random forest with per-tree variance
# ---------------------------------------------------------------------------

class RandomForestModel(UncertaintyModel):
    '''Random forest regressor using per-tree variance as uncertainty.

    The forest's uncertainty is the standard deviation across its constituent
    trees' predictions, which is the standard ensemble uncertainty estimator.
    '''

    name = 'random_forest'

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **estimator_kwargs,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs
        self._model: Optional[RandomForestRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.estimator_kwargs,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('RandomForestModel must be fitted first')
        return self._model.predict(np.asarray(X, dtype=float))

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError('RandomForestModel must be fitted first')
        X = np.asarray(X, dtype=float)
        tree_preds = np.stack(
            [tree.predict(X) for tree in self._model.estimators_], axis=0
        )
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0, ddof=1) if tree_preds.shape[0] > 1 else np.zeros_like(mean)
        return mean, std


# ---------------------------------------------------------------------------
# XGBoost with bagged uncertainty
# ---------------------------------------------------------------------------

class XGBoostModel(UncertaintyModel):
    '''XGBoost regressor with bagged uncertainty.

    XGBoost does not expose per-tree variance in a form useful for predictive
    uncertainty, so we bag ``n_bootstrap`` XGBoost models on resampled
    training sets and report the mean and std across the bag.
    '''

    name = 'xgboost'

    def __init__(
        self,
        n_bootstrap: int = 10,
        random_state: Optional[int] = None,
        **estimator_kwargs,
    ) -> None:
        if not _HAS_XGBOOST:  # pragma: no cover
            raise ImportError(
                'xgboost is required to use XGBoostModel. Install it with '
                "`pip install 'uncertainty-imputation[xgboost]'`."
            )
        self.n_bootstrap = int(n_bootstrap)
        self.random_state = random_state
        # Sensible defaults for imputation: small trees, modest depth.
        defaults = dict(n_estimators=100, max_depth=4, learning_rate=0.1,
                        verbosity=0, tree_method='hist')
        defaults.update(estimator_kwargs)
        self.estimator_kwargs = defaults
        self._models: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostModel':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31 - 1, size=self.n_bootstrap)
        n = X.shape[0]
        self._models = []
        for seed in seeds:
            idx = rng.integers(0, n, size=n)
            model = XGBRegressor(random_state=int(seed), **self.estimator_kwargs)
            model.fit(X[idx], y[idx])
            self._models.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(X)
        return mean

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._models:
            raise RuntimeError('XGBoostModel must be fitted first')
        X = np.asarray(X, dtype=float)
        preds = np.stack([m.predict(X) for m in self._models], axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1) if self.n_bootstrap > 1 else np.zeros_like(mean)
        return mean, std


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[UncertaintyModel]] = {
    'linear': LinearRegressionModel,
    'linear_regression': LinearRegressionModel,
    'tree': DecisionTreeModel,
    'decision_tree': DecisionTreeModel,
    'rf': RandomForestModel,
    'random_forest': RandomForestModel,
}

if _HAS_XGBOOST:
    _MODEL_REGISTRY['xgb'] = XGBoostModel
    _MODEL_REGISTRY['xgboost'] = XGBoostModel


def get_model(
    model: 'str | UncertaintyModel',
    random_state: Optional[int] = None,
    **kwargs,
) -> UncertaintyModel:
    '''Resolve a model from a string alias or instance.

    Parameters
    ----------
    model : str or UncertaintyModel
        Alias (``'linear'``, ``'decision_tree'``, ``'random_forest'``,
        ``'xgboost'``) or an already-constructed :class:`UncertaintyModel`.
    random_state : int, optional
        Seed forwarded to the model constructor when ``model`` is a string.
    **kwargs
        Extra estimator keyword arguments.
    '''
    if isinstance(model, UncertaintyModel):
        return model
    if not isinstance(model, str):
        raise TypeError(
            'model must be a string alias or UncertaintyModel instance, '
            f'got {type(model).__name__}'
        )
    key = model.lower()
    if key not in _MODEL_REGISTRY:
        available = sorted(set(_MODEL_REGISTRY))
        raise ValueError(f'Unknown model {model!r}. Available: {available}')
    return _MODEL_REGISTRY[key](random_state=random_state, **kwargs)


def register_model(name: str, cls: type[UncertaintyModel]) -> None:
    '''Register a custom model under a string alias.'''
    if not issubclass(cls, UncertaintyModel):
        raise TypeError('cls must be a subclass of UncertaintyModel')
    _MODEL_REGISTRY[name.lower()] = cls


__all__ = [
    'UncertaintyModel',
    'LinearRegressionModel',
    'DecisionTreeModel',
    'RandomForestModel',
    'XGBoostModel',
    'get_model',
    'register_model',
]
