"""MNAR (Missing Not At Random) amputation.

Under MNAR, the probability of missingness depends on the *value* of the
column itself. This is the hardest missingness mechanism because the
missing data mechanism cannot be ignored for valid inference. We implement
a simple value-threshold MNAR where rows with the largest (or smallest)
values in a target column have the highest chance of being made missing.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .base import BaseAmputer


class MNARAmputer(BaseAmputer):
    '''Missing Not At Random amputation based on value thresholds.

    Parameters
    ----------
    missing_rate : float, default 0.2
        Target mean missing rate per eligible column.
    columns : sequence of int or str, optional
        Columns to ampute.
    direction : {'upper', 'lower', 'extremes'}, default ``'upper'``
        Where to concentrate missingness. ``'upper'`` hides the largest
        values, ``'lower'`` the smallest, ``'extremes'`` both tails.
    strength : float, default 1.0
        In ``[0, 1]``. ``1.0`` makes missingness deterministic (the
        ``missing_rate`` most extreme values are masked). Values below 1
        mix in random noise so the mechanism is stochastic.
    random_state : int, optional
        Seed for reproducibility.
    '''

    def __init__(
        self,
        missing_rate: float = 0.2,
        columns: Optional[Sequence[Union[int, str]]] = None,
        direction: str = 'upper',
        strength: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            missing_rate=missing_rate,
            columns=columns,
            random_state=random_state,
        )
        if direction not in ('upper', 'lower', 'extremes'):
            raise ValueError(
                "direction must be 'upper', 'lower' or 'extremes', got "
                f'{direction!r}'
            )
        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                f'strength must be in [0, 1], got {strength}'
            )
        self.direction = direction
        self.strength = float(strength)

    def _generate_mask(
        self,
        arr: np.ndarray,
        cols: list[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        mask = np.zeros_like(arr, dtype=bool)
        if self.missing_rate == 0.0 or not cols:
            return mask

        n = arr.shape[0]
        for j in cols:
            values = arr[:, j].astype(float)
            # Rank values in [0, 1] so that the ranking is scale-free.
            order = np.argsort(values, kind='stable')
            ranks = np.empty(n, dtype=float)
            ranks[order] = np.arange(n) / max(n - 1, 1)

            if self.direction == 'upper':
                score = ranks
            elif self.direction == 'lower':
                score = 1.0 - ranks
            else:  # extremes
                score = np.abs(ranks - 0.5) * 2.0

            noise = rng.random(size=n)
            combined = self.strength * score + (1.0 - self.strength) * noise
            threshold = np.quantile(combined, 1.0 - self.missing_rate)
            mask[:, j] = combined > threshold

        return mask


__all__ = ['MNARAmputer']
