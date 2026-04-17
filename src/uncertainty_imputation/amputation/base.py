"""Base class for amputation (missing data simulation)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame]


class BaseAmputer(ABC):
    '''Abstract base class for amputers.

    An amputer introduces missing values into a complete dataset. Subclasses
    implement :meth:`_generate_mask` which returns a boolean mask the same
    shape as ``X`` (True where values should be set to NaN).
    '''

    def __init__(
        self,
        missing_rate: float = 0.2,
        columns: Optional[Sequence[Union[int, str]]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError(
                f'missing_rate must be in [0, 1], got {missing_rate}'
            )
        self.missing_rate = float(missing_rate)
        self.columns = list(columns) if columns is not None else None
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_columns(
        self, X: ArrayLike, arr_shape: tuple[int, int]
    ) -> list[int]:
        '''Resolve ``self.columns`` to integer indices.'''
        if self.columns is None:
            return list(range(arr_shape[1]))
        if isinstance(X, pd.DataFrame):
            col_list = list(X.columns)
            resolved = []
            for c in self.columns:
                if isinstance(c, int):
                    resolved.append(c)
                elif c in col_list:
                    resolved.append(col_list.index(c))
                else:
                    raise KeyError(f'Column {c!r} not found in DataFrame')
            return resolved
        # numpy: columns must be ints
        for c in self.columns:
            if not isinstance(c, (int, np.integer)):
                raise TypeError(
                    'columns must be a sequence of ints when X is a numpy '
                    f'array; got {type(c).__name__}'
                )
        return [int(c) for c in self.columns]

    def _to_array(
        self, X: ArrayLike
    ) -> tuple[np.ndarray, Optional[list[str]], Optional[pd.Index]]:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float, copy=True), list(X.columns), X.index
        if isinstance(X, np.ndarray):
            return np.array(X, dtype=float, copy=True), None, None
        raise TypeError(
            'X must be a numpy array or pandas DataFrame, got '
            f'{type(X).__name__}'
        )

    def _wrap(
        self,
        arr: np.ndarray,
        columns: Optional[list[str]],
        index: Optional[pd.Index],
    ) -> ArrayLike:
        if columns is not None:
            return pd.DataFrame(arr, columns=columns, index=index)
        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @abstractmethod
    def _generate_mask(
        self,
        arr: np.ndarray,
        cols: list[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        '''Return a boolean mask (True = make missing).'''

    def ampute(
        self, X: ArrayLike, return_mask: bool = False
    ) -> Union[ArrayLike, tuple[ArrayLike, np.ndarray]]:
        '''Introduce missing values into ``X`` and return the amputed copy.

        Parameters
        ----------
        X : np.ndarray or pandas.DataFrame
            Input complete dataset.
        return_mask : bool, default False
            If True, return ``(X_amputed, mask)``.

        Returns
        -------
        X_amputed, optionally with the boolean mask.
        '''
        arr, columns, index = self._to_array(X)
        cols = self._resolve_columns(X, arr.shape)
        rng = np.random.default_rng(self.random_state)
        mask = self._generate_mask(arr, cols, rng)
        arr_amputed = arr.copy()
        arr_amputed[mask] = np.nan
        wrapped = self._wrap(arr_amputed, columns, index)
        if return_mask:
            return wrapped, mask
        return wrapped

    # Convenience alias
    def __call__(self, X: ArrayLike, return_mask: bool = False):
        return self.ampute(X, return_mask=return_mask)


__all__ = ['BaseAmputer', 'ArrayLike']
