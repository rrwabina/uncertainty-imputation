"""Domain-specified predictors: only use certain columns to impute others.

Real-world datasets often have known relationships between variables: BMI
depends on height and weight; blood pressure depends on age and BMI; etc.
Instead of blindly using every column to impute every other column, the
``predictor_columns`` parameter lets you encode these relationships.

Run with::

    python examples/05_predictor_columns.py
"""

import numpy as np
import pandas as pd

from uncertainty_imputation import MCARAmputer, UncertaintyImputer
from uncertainty_imputation.utils.metrics import rmse


def main() -> None:
    # Build a synthetic clinical-style dataset where we know the causal
    # structure: bmi = f(height, weight), sbp = f(age, bmi).
    rng = np.random.default_rng(0)
    n = 400
    age = rng.uniform(20, 80, size=n)
    height = rng.normal(170, 10, size=n)          # cm
    weight = rng.normal(70, 15, size=n)           # kg
    bmi = weight / (height / 100) ** 2 + 0.5 * rng.normal(size=n)
    sbp = 0.5 * age + 1.5 * bmi + 80 + 2.0 * rng.normal(size=n)
    # irrelevant variable — shouldn't be used as a predictor for anything
    noise_col = rng.normal(size=n)

    X_true = pd.DataFrame({
        'age': age,
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'sbp': sbp,
        'noise': noise_col,
    })
    X_missing = MCARAmputer(missing_rate=0.25, random_state=0).ampute(X_true)
    mask = X_missing.isna().to_numpy()

    # -------------------------------------------------------------------
    # Compare: all-predictors default vs. domain-specified predictors
    # -------------------------------------------------------------------
    imp_all = UncertaintyImputer(
        model='linear',
        acquisition='us',
        max_iter=5,
        random_state=42,
        update_rule='mean',  # hold update rule fixed to isolate predictor effect
    ).fit_transform(X_missing)

    # Encode domain knowledge: bmi <- height, weight; sbp <- age, bmi;
    # age, height, weight don't depend on other features we've captured;
    # the 'noise' column is not used to predict anything.
    imp_domain = UncertaintyImputer(
        model='linear',
        acquisition='us',
        max_iter=5,
        random_state=42,
        update_rule='mean',
        predictor_columns={
            'bmi':    ['height', 'weight'],
            'sbp':    ['age', 'bmi'],
            'age':    ['sbp'],             # age is correlated with sbp
            'height': ['weight', 'bmi'],   # height correlates with weight, bmi
            'weight': ['height', 'bmi'],
            # 'noise' not listed -> falls back to 'all other columns',
            # but since it's truly random, using all columns won't help it
        },
    ).fit_transform(X_missing)

    # -------------------------------------------------------------------
    # Compare RMSE per column
    # -------------------------------------------------------------------
    print(f'Dataset: {X_true.shape}  Missing: {mask.sum()} cells '
          f'({100 * mask.mean():.1f}%)\n')
    print(f"{'column':<10} {'RMSE (all)':>12} {'RMSE (domain)':>15} "
          f"{'delta':>8}")
    print('-' * 50)
    for j, col in enumerate(X_true.columns):
        col_mask = mask[:, j]
        if not col_mask.any():
            continue
        # Per-column mask: only this column's missing cells
        m = np.zeros_like(mask)
        m[col_mask, j] = True
        r_all = rmse(X_true, imp_all, m)
        r_dom = rmse(X_true, imp_domain, m)
        print(f"{col:<10} {r_all:>12.4f} {r_dom:>15.4f} "
              f"{r_dom - r_all:>+8.4f}")

    print('\nOverall:')
    print(f'  all-predictors    RMSE = {rmse(X_true, imp_all, mask):.4f}')
    print(f'  domain predictors RMSE = {rmse(X_true, imp_domain, mask):.4f}')


if __name__ == '__main__':
    main()
