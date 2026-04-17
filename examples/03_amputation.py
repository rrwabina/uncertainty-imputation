"""Amputation mechanisms: MCAR, MAR, MNAR.

Demonstrates the three missingness mechanisms and how each biases the
observed data distribution differently.

Run with::

    python examples/03_amputation.py
"""

import numpy as np
import pandas as pd

from uncertainty_imputation import MARAmputer, MCARAmputer, MNARAmputer


def main() -> None:
    # Build a complete dataset: y correlates with x.
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    y = 0.6 * x + 0.5 * rng.normal(size=n)
    X = pd.DataFrame({'x': x, 'y': y})

    print(f'Original dataset: {X.shape}')
    print(f'  y mean={X.y.mean():.3f}  std={X.y.std():.3f}\n')

    # -------------------------------------------------------------------
    # MCAR — missingness independent of everything
    # -------------------------------------------------------------------
    X_mcar, mask_mcar = MCARAmputer(
        missing_rate=0.3, columns=['y'], random_state=0,
    ).ampute(X, return_mask=True)
    print('MCAR:')
    print(f'  y missing rate: {mask_mcar[:, 1].mean():.3f}')
    print(f'  y | observed  : mean={X_mcar.y.mean():.3f} '
          f'std={X_mcar.y.std():.3f}')
    print('  (the observed distribution of y is essentially unchanged)\n')

    # -------------------------------------------------------------------
    # MAR — missingness in y depends on the observed x
    # -------------------------------------------------------------------
    X_mar, mask_mar = MARAmputer(
        missing_rate=0.3, columns=['y'], predictor_columns=['x'],
        strength=4.0, random_state=0,
    ).ampute(X, return_mask=True)
    mar_mask_col = mask_mar[:, 1]
    print('MAR (missingness of y driven by x):')
    print(f'  y missing rate: {mar_mask_col.mean():.3f}')
    print(f'  x | y missing : mean={x[mar_mask_col].mean():+.3f}')
    print(f'  x | y present : mean={x[~mar_mask_col].mean():+.3f}')
    print('  (x has a different mean depending on whether y is missing)\n')

    # -------------------------------------------------------------------
    # MNAR — missingness depends on y itself
    # -------------------------------------------------------------------
    X_mnar, mask_mnar = MNARAmputer(
        missing_rate=0.3, columns=['y'], direction='upper',
        strength=1.0, random_state=0,
    ).ampute(X, return_mask=True)
    mnar_mask_col = mask_mnar[:, 1]
    print('MNAR (upper-tail of y hidden):')
    print(f'  y missing rate: {mnar_mask_col.mean():.3f}')
    print(f'  y | observed  : mean={X_mnar.y.mean():+.3f}')
    print(f'  y | hidden    : mean={y[mnar_mask_col].mean():+.3f}')
    print('  (the observed y is systematically smaller than the hidden y)')


if __name__ == '__main__':
    main()
