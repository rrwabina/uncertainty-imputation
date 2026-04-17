"""MCAR (Missing Completely At Random) amputation."""

from __future__ import annotations

import numpy as np

from .base import BaseAmputer


class MCARAmputer(BaseAmputer):
    '''Missing Completely At Random amputation.

    Each eligible cell is independently masked with probability equal to
    ``missing_rate``. The resulting missingness is independent of any
    observed or unobserved variables.

    Parameters
    ----------
    missing_rate : float, default 0.2
        Per-cell probability of being made missing (on the eligible columns).
    columns : sequence of int or str, optional
        Columns in which to induce missingness. If ``None``, all columns are
        eligible.
    random_state : int, optional
        Seed for reproducibility.
    '''

    def _generate_mask(
        self,
        arr: np.ndarray,
        cols: list[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        mask = np.zeros_like(arr, dtype=bool)
        if self.missing_rate == 0.0 or not cols:
            return mask
        draws = rng.random(size=(arr.shape[0], len(cols)))
        mask[:, cols] = draws < self.missing_rate
        return mask


__all__ = ['MCARAmputer']
