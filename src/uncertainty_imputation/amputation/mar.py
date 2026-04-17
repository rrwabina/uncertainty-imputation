"""MAR (Missing At Random) amputation.

Under MAR, the probability of missingness in a target column depends on the
*observed* values of other columns in the dataset (but not on the missing
values themselves). We implement this via a logistic model of a linear
score computed from a set of predictor columns.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import BaseAmputer


class MARAmputer(BaseAmputer):
    '''Missing At Random amputation.

    Parameters
    ----------
    missing_rate : float, default 0.2
        Target mean missing rate (per eligible column). A sigmoid-transformed
        linear score of the predictor columns is thresholded so that the
        empirical rate matches ``missing_rate``.
    columns : sequence of int or str, optional
        Target columns in which to induce missingness. If ``None``, all
        columns are eligible targets.
    predictor_columns : sequence of int or str, optional
        Observed columns used to determine missingness probabilities. If
        ``None``, every column that is *not* the current target is used as
        a predictor.
    strength : float, default 3.0
        Slope of the logistic function; larger values make missingness more
        strongly dependent on the predictors.
    random_state : int, optional
        Seed for reproducibility.

    Notes
    -----
    The score for each target column is a random linear combination of the
    (standardised) predictor columns. The threshold is chosen per column so
    the empirical missing fraction matches ``missing_rate``, which gives a
    well-controlled dependence on observed values — the defining feature of
    MAR.
    '''

    def __init__(
        self,
        missing_rate: float = 0.2,
        columns: Optional[Sequence[Union[int, str]]] = None,
        predictor_columns: Optional[Sequence[Union[int, str]]] = None,
        strength: float = 3.0,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            missing_rate=missing_rate,
            columns=columns,
            random_state=random_state,
        )
        self.predictor_columns = (
            list(predictor_columns) if predictor_columns is not None else None
        )
        self.strength = float(strength)

    def _resolve_predictor_columns(
        self, X, arr_shape: tuple[int, int]
    ) -> Optional[list[int]]:
        if self.predictor_columns is None:
            return None
        if isinstance(X, pd.DataFrame):
            col_list = list(X.columns)
            resolved = []
            for c in self.predictor_columns:
                if isinstance(c, int):
                    resolved.append(c)
                elif c in col_list:
                    resolved.append(col_list.index(c))
                else:
                    raise KeyError(f'Predictor column {c!r} not found')
            return resolved
        for c in self.predictor_columns:
            if not isinstance(c, (int, np.integer)):
                raise TypeError(
                    'predictor_columns must be a sequence of ints when X '
                    f'is a numpy array; got {type(c).__name__}'
                )
        return [int(c) for c in self.predictor_columns]

    def _generate_mask(
        self,
        arr: np.ndarray,
        cols: list[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        mask = np.zeros_like(arr, dtype=bool)
        if self.missing_rate == 0.0 or not cols:
            return mask

        n_rows = arr.shape[0]

        for j in cols:
            # Determine predictor columns for this target.
            if self.predictor_columns is not None:
                pred_ids = [p for p in self._predictor_idx_cache if p != j]
            else:
                pred_ids = [k for k in range(arr.shape[1]) if k != j]
            if not pred_ids:
                # Fall back to MCAR if no predictors are available.
                draws = rng.random(size=n_rows)
                mask[:, j] = draws < self.missing_rate
                continue

            predictors = arr[:, pred_ids]
            # Standardise predictors (NaNs treated as zero-mean / unit-var)
            col_mean = np.nanmean(predictors, axis=0)
            col_std = np.nanstd(predictors, axis=0)
            col_std = np.where(col_std < 1e-12, 1.0, col_std)
            standardised = (predictors - col_mean) / col_std
            # Replace remaining NaNs with 0 so they contribute neutrally.
            standardised = np.nan_to_num(standardised, nan=0.0)

            # Random linear combination + logistic transform.
            weights = rng.normal(loc=0.0, scale=1.0, size=len(pred_ids))
            weights = weights / max(np.linalg.norm(weights), 1e-12)
            logits = self.strength * standardised @ weights
            # Pick threshold so the empirical missing rate matches target.
            threshold = np.quantile(logits, 1.0 - self.missing_rate)
            # Add small jitter to break ties deterministically.
            jitter = rng.normal(loc=0.0, scale=1e-6, size=n_rows)
            mask[:, j] = (logits + jitter) > threshold

        return mask

    def ampute(self, X, return_mask: bool = False):  # type: ignore[override]
        # Pre-resolve predictor indices once up front.
        arr, columns, index = self._to_array(X)
        self._predictor_idx_cache = (
            self._resolve_predictor_columns(X, arr.shape) or []
        )
        return super().ampute(X, return_mask=return_mask)


__all__ = ['MARAmputer']
