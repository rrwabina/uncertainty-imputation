"""Benchmark: uncertainty imputer vs mean/MICE baselines.

Runs every ``(model, acquisition)`` combination supported by the library on
an amputed dataset and compares it to mean imputation and scikit-learn's
IterativeImputer (MICE).

Run with::

    python examples/02_benchmark.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from uncertainty_imputation import (
    MCARAmputer,
    MeanImputer,
    MICEImputer,
    UncertaintyImputer,
)
from uncertainty_imputation.utils.metrics import mae, rmse

try:
    import xgboost  # noqa: F401

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def make_dataset(n: int = 500, seed: int = 0) -> pd.DataFrame:
    '''Build a moderately correlated numeric dataset with nonlinear signal.'''
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.7 * x1 + 0.4 * rng.normal(size=n)
    x3 = np.tanh(x1) + 0.3 * rng.normal(size=n)
    x4 = -0.5 * x2 + 0.3 * rng.normal(size=n)
    x5 = rng.normal(size=n)
    return pd.DataFrame({
        'f1': x1, 'f2': x2, 'f3': x3, 'f4': x4, 'f5': x5,
    })


def _time_imputer(imputer, X_missing) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    out = imputer.fit_transform(X_missing)
    return out, time.perf_counter() - start


def main() -> None:
    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------
    X_true = make_dataset(n=500, seed=0)
    X_missing = MCARAmputer(missing_rate=0.25, random_state=0).ampute(X_true)
    mask = np.isnan(X_missing.to_numpy())
    print(f'Dataset: {X_true.shape}  Missing: {mask.sum()} cells '
          f'({100 * mask.mean():.1f}%)')

    # -------------------------------------------------------------------
    # Build the grid of imputers
    # -------------------------------------------------------------------
    models = ['linear', 'decision_tree', 'random_forest']
    if _HAS_XGB:
        models.append('xgboost')
    acquisitions = ['us', 'pi', 'ei']

    runs: list[dict] = []

    # Baselines
    for name, imputer in [
        ('MeanImputer', MeanImputer()),
        ('MICEImputer (sklearn)', MICEImputer(max_iter=5, random_state=42)),
    ]:
        out, elapsed = _time_imputer(imputer, X_missing)
        runs.append({
            'imputer': name,
            'model': '—',
            'acquisition': '—',
            'rmse': rmse(X_true, out, mask),
            'mae': mae(X_true, out, mask),
            'seconds': elapsed,
        })

    # Uncertainty imputer variants
    for model in models:
        model_kwargs = {}
        if model == 'random_forest':
            model_kwargs = {'n_estimators': 30}
        elif model == 'xgboost':
            model_kwargs = {'n_bootstrap': 5, 'n_estimators': 50}
        elif model in ('linear', 'decision_tree'):
            model_kwargs = {'n_bootstrap': 10}

        for acq in acquisitions:
            imputer = UncertaintyImputer(
                model=model,
                acquisition=acq,
                max_iter=5,
                random_state=42,
                model_kwargs=model_kwargs,
            )
            out, elapsed = _time_imputer(imputer, X_missing)
            runs.append({
                'imputer': 'UncertaintyImputer',
                'model': model,
                'acquisition': acq,
                'rmse': rmse(X_true, out, mask),
                'mae': mae(X_true, out, mask),
                'seconds': elapsed,
            })

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    df = pd.DataFrame(runs).sort_values('rmse').reset_index(drop=True)
    df['rmse'] = df['rmse'].round(4)
    df['mae'] = df['mae'].round(4)
    df['seconds'] = df['seconds'].round(2)
    print('\nResults (sorted by RMSE, lower is better):\n')
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest: {best['imputer']} / model={best['model']} / "
          f"acq={best['acquisition']} -> RMSE {best['rmse']}")


if __name__ == '__main__':
    main()
