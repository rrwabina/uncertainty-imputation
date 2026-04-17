"""Quickstart: impute missing values in 30 seconds.

Run from the repo root with::

    python examples/01_quickstart.py
"""

import numpy as np
import pandas as pd

from uncertainty_imputation import MCARAmputer, UncertaintyImputer
from uncertainty_imputation.utils.metrics import evaluate_imputation


def main() -> None:
    # 1. Build a complete toy dataset (four correlated numeric features).
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.normal(size=n)
    x2 = 0.8 * x1 + 0.3 * rng.normal(size=n)
    x3 = -0.5 * x1 + 0.4 * rng.normal(size=n)
    x4 = rng.normal(size=n)
    X_true = pd.DataFrame({
        'age': x1,
        'bp': x2,
        'chol': x3,
        'bmi': x4,
    })

    # 2. Simulate 25% missingness, completely at random.
    X_missing = MCARAmputer(missing_rate=0.25, random_state=0).ampute(X_true)
    mask = np.isnan(X_missing.to_numpy())
    print(f'Introduced {mask.sum()} missing cells '
          f'({100 * mask.mean():.1f}% of the matrix)')

    # 3. Impute with a random forest + expected improvement acquisition.
    imputer = UncertaintyImputer(
        model='random_forest',
        acquisition='ei',
        max_iter=5,
        random_state=42,
        model_kwargs={'n_estimators': 50},
        verbose=True,
    )
    X_imputed = imputer.fit_transform(X_missing)

    # 4. Evaluate against the ground truth.
    metrics = evaluate_imputation(X_true, X_imputed, mask)
    print('\nImputation quality:')
    for k, v in metrics.items():
        print(f'  {k:>10}: {v:.4f}' if isinstance(v, float) else f'  {k:>10}: {v}')

    # 5. Inspect recorded uncertainty (acquisition scores per missing cell).
    print('\nMean acquisition score per column:')
    for col_idx, scores in imputer.uncertainty_.items():
        print(f'  {X_missing.columns[col_idx]:>6}: '
              f'mean={scores.mean():.3f}  max={scores.max():.3f}')


if __name__ == '__main__':
    main()
