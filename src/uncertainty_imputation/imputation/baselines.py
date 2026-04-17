"""Baseline imputers for benchmarking against :class:`UncertaintyImputer`.

These wrappers give the library a like-for-like API for common baselines so
evaluation pipelines can swap them in with zero friction.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..imputation.base import ArrayLike, BaseImputer


class MeanImputer(BaseImputer):
    '''Simple per-column mean imputer.'''

    def __init__(self, strategy: str = 'mean') -> None:
        super().__init__()
        if strategy not in ('mean', 'median'):
            raise ValueError(
                f"strategy must be 'mean' or 'median', got {strategy!r}"
            )
        self.strategy = strategy
        self._fill_values_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y=None) -> 'MeanImputer':
        arr, _, _ = self._to_array(X)
        if self.strategy == 'mean':
            self._fill_values_ = np.nanmean(arr, axis=0)
        else:
            self._fill_values_ = np.nanmedian(arr, axis=0)
        # Any column that is entirely NaN gets filled with 0.
        self._fill_values_ = np.nan_to_num(self._fill_values_, nan=0.0)
        self._n_features = arr.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        self._check_is_fitted()
        arr, columns, index = self._to_array(X)
        if arr.shape[1] != self._n_features:
            raise ValueError(
                f'X has {arr.shape[1]} features but imputer was fit with '
                f'{self._n_features}'
            )
        mask = np.isnan(arr)
        out = arr.copy()
        for j in range(arr.shape[1]):
            col_mask = mask[:, j]
            if col_mask.any():
                out[col_mask, j] = self._fill_values_[j]
        return self._wrap_output(out, columns, index)


class MICEImputer(BaseImputer):
    '''Thin wrapper around ``sklearn.impute.IterativeImputer`` (MICE-style).

    Provided as a standard baseline. All keyword arguments are forwarded to
    :class:`sklearn.impute.IterativeImputer`.
    '''

    def __init__(
        self,
        max_iter: int = 10,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Lazy import so sklearn's experimental flag is only triggered when
        # this class is actually used.
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.kwargs = dict(kwargs)
        self._imputer = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.kwargs,
        )

    def fit(self, X: ArrayLike, y=None) -> 'MICEImputer':
        arr, _, _ = self._to_array(X)
        self._imputer.fit(arr)
        self._n_features = arr.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        self._check_is_fitted()
        arr, columns, index = self._to_array(X)
        out = self._imputer.transform(arr)
        return self._wrap_output(out, columns, index)


__all__ = ['MeanImputer', 'MICEImputer']
