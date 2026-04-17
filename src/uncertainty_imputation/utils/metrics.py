"""Evaluation metrics for imputation quality.

All metrics operate on three aligned arrays:

* ``X_true`` — ground-truth complete data
* ``X_imputed`` — imputed output from a model
* ``mask`` — boolean mask where True marks originally-missing cells

Metrics are computed only over the masked (imputed) cells.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame]


def _to_array(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def _prepare(
    X_true: ArrayLike, X_imputed: ArrayLike, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_true_arr = _to_array(X_true)
    X_imp_arr = _to_array(X_imputed)
    mask = np.asarray(mask, dtype=bool)
    if X_true_arr.shape != X_imp_arr.shape:
        raise ValueError(
            f'Shape mismatch: X_true {X_true_arr.shape} vs '
            f'X_imputed {X_imp_arr.shape}'
        )
    if mask.shape != X_true_arr.shape:
        raise ValueError(
            f'Mask shape {mask.shape} does not match data shape '
            f'{X_true_arr.shape}'
        )
    if not mask.any():
        raise ValueError('mask contains no missing cells to evaluate')
    return X_true_arr, X_imp_arr, mask


def rmse(X_true: ArrayLike, X_imputed: ArrayLike, mask: np.ndarray) -> float:
    '''Root Mean Squared Error over imputed cells.'''
    X_true_arr, X_imp_arr, mask = _prepare(X_true, X_imputed, mask)
    diff = X_true_arr[mask] - X_imp_arr[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(X_true: ArrayLike, X_imputed: ArrayLike, mask: np.ndarray) -> float:
    '''Mean Absolute Error over imputed cells.'''
    X_true_arr, X_imp_arr, mask = _prepare(X_true, X_imputed, mask)
    return float(np.mean(np.abs(X_true_arr[mask] - X_imp_arr[mask])))


def normalized_rmse(
    X_true: ArrayLike, X_imputed: ArrayLike, mask: np.ndarray
) -> float:
    '''RMSE normalised by the std of ``X_true`` over the same cells.

    Useful when comparing imputation quality across features with very
    different scales.
    '''
    X_true_arr, X_imp_arr, mask = _prepare(X_true, X_imputed, mask)
    denom = np.std(X_true_arr[mask])
    if denom < 1e-12:
        return float('nan')
    return float(rmse(X_true_arr, X_imp_arr, mask) / denom)


def per_column_rmse(
    X_true: ArrayLike,
    X_imputed: ArrayLike,
    mask: np.ndarray,
) -> np.ndarray:
    '''RMSE computed separately for each column.

    Returns an array of length ``n_features``. Columns with no missing cells
    have value ``NaN``.
    '''
    X_true_arr, X_imp_arr, mask = _prepare(X_true, X_imputed, mask)
    n_features = X_true_arr.shape[1]
    out = np.full(n_features, np.nan)
    for j in range(n_features):
        col_mask = mask[:, j]
        if col_mask.any():
            diff = X_true_arr[col_mask, j] - X_imp_arr[col_mask, j]
            out[j] = float(np.sqrt(np.mean(diff ** 2)))
    return out


def evaluate_imputation(
    X_true: ArrayLike,
    X_imputed: ArrayLike,
    mask: np.ndarray,
) -> dict:
    '''Return a dict of all standard metrics in one call.'''
    return dict(
        rmse=rmse(X_true, X_imputed, mask),
        mae=mae(X_true, X_imputed, mask),
        nrmse=normalized_rmse(X_true, X_imputed, mask),
        n_imputed=int(np.asarray(mask, dtype=bool).sum()),
    )


__all__ = [
    'rmse',
    'mae',
    'normalized_rmse',
    'per_column_rmse',
    'evaluate_imputation',
]
