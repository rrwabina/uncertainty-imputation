"""Base imputer class providing a scikit-learn-compatible interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame]


class BaseImputer(ABC):
    '''Abstract base class for imputers.

    Subclasses implement :meth:`fit` and :meth:`transform`. The base class
    handles input validation and round-tripping between pandas DataFrames and
    numpy arrays.
    '''

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._columns: Optional[list[str]] = None
        self._n_features: Optional[int] = None

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _to_array(self, X: ArrayLike) -> tuple[np.ndarray, Optional[list[str]], Optional[pd.Index]]:
        '''Convert X to a float64 array, returning (array, columns, index).'''
        if isinstance(X, pd.DataFrame):
            columns = list(X.columns)
            index = X.index
            arr = X.to_numpy(dtype=float, copy=True)
        elif isinstance(X, np.ndarray):
            columns = None
            index = None
            arr = np.array(X, dtype=float, copy=True)
        else:
            raise TypeError(
                'X must be a numpy array or pandas DataFrame, got '
                f'{type(X).__name__}'
            )
        if arr.ndim != 2:
            raise ValueError(f'X must be 2D, got shape {arr.shape}')
        return arr, columns, index

    def _wrap_output(
        self,
        arr: np.ndarray,
        columns: Optional[list[str]],
        index: Optional[pd.Index],
    ) -> ArrayLike:
        if columns is not None:
            return pd.DataFrame(arr, columns=columns, index=index)
        return arr

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f'{type(self).__name__} must be fitted before calling transform. '
                'Call .fit(X) first, or use .fit_transform(X).'
            )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, X: ArrayLike, y=None) -> 'BaseImputer':
        '''Fit the imputer on ``X``.'''

    @abstractmethod
    def transform(self, X: ArrayLike) -> ArrayLike:
        '''Impute missing values in ``X``.'''

    def fit_transform(self, X: ArrayLike, y=None) -> ArrayLike:
        '''Fit and transform in one call.

        Subclasses can override this for efficiency (the default does one pass
        of work during :meth:`fit` and re-uses state in :meth:`transform`).
        '''
        return self.fit(X, y=y).transform(X)


__all__ = ['BaseImputer', 'ArrayLike']
