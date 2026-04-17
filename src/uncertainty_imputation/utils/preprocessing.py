"""Preprocessing utilities for the imputation library."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame]


def get_missing_mask(X: ArrayLike) -> np.ndarray:
    '''Return a boolean mask where True indicates a missing value.

    Accepts numpy arrays and pandas DataFrames; missing values are detected
    via ``np.isnan`` (for numeric arrays) or ``pandas.isna`` (for
    DataFrames, including object columns).
    '''
    if isinstance(X, pd.DataFrame):
        return X.isna().to_numpy()
    arr = np.asarray(X)
    if np.issubdtype(arr.dtype, np.floating):
        return np.isnan(arr)
    # Non-float arrays: coerce through pandas to catch object NaNs.
    return pd.isna(arr)


def missing_summary(X: ArrayLike) -> pd.DataFrame:
    '''Return a per-column summary of missingness.

    Columns of the returned DataFrame:

    * ``n_missing`` — number of missing entries
    * ``missing_rate`` — fraction of missing entries
    * ``dtype`` — dtype of the column (for DataFrames)
    '''
    mask = get_missing_mask(X)
    n_rows = mask.shape[0]
    n_missing = mask.sum(axis=0)
    rate = n_missing / max(n_rows, 1)
    if isinstance(X, pd.DataFrame):
        idx = X.columns
        dtypes = X.dtypes.astype(str).tolist()
    else:
        idx = pd.Index(range(mask.shape[1]), name='column')
        dtypes = [str(np.asarray(X).dtype)] * mask.shape[1]
    return pd.DataFrame(
        dict(n_missing=n_missing, missing_rate=rate, dtype=dtypes),
        index=idx,
    )


def standardize(X: ArrayLike, *, return_stats: bool = False):
    '''Standardise columns to zero mean and unit variance, ignoring NaNs.

    Parameters
    ----------
    X : array-like
        Input matrix.
    return_stats : bool, default False
        If True, also return the ``(mean, std)`` arrays used so the
        transform can be inverted later.
    '''
    if isinstance(X, pd.DataFrame):
        mean = X.mean(skipna=True)
        std = X.std(skipna=True).replace(0.0, 1.0)
        out = (X - mean) / std
        if return_stats:
            return out, mean.to_numpy(), std.to_numpy()
        return out
    arr = np.asarray(X, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    out = (arr - mean) / std
    if return_stats:
        return out, mean, std
    return out


def unstandardize(
    X: ArrayLike, mean: np.ndarray, std: np.ndarray
) -> ArrayLike:
    '''Invert :func:`standardize` using stored mean/std arrays.'''
    if isinstance(X, pd.DataFrame):
        return X * std + mean
    return np.asarray(X, dtype=float) * std + mean


__all__ = [
    'get_missing_mask',
    'missing_summary',
    'standardize',
    'unstandardize',
]
