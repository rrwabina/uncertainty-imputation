# uncertainty-imputation

**Uncertainty-aware iterative imputation for tabular data, guided by acquisition functions.**

A production-ready Python library implementing the framework from:

> *Uncertainty-aware approach for multiple imputation using conventional and machine learning models: a real-world data study.*

The library iteratively imputes missing values in tabular datasets like MICE but trains uncertainty-aware models at each step and records a per-cell **acquisition score** (Uncertainty Sampling, Probability of Improvement, or Expected Improvement) that downstream code can use to weight, prioritise, or audit individual imputations.

A scikit-learn-style API, support for pandas DataFrames and numpy arrays, a pluggable registry for custom models and acquisition functions, and a built-in amputation module (MCAR / MAR / MNAR) for generating missingness in controlled experiments.

---

## Installation

```bash
pip install uncertainty-imputation               # core
pip install 'uncertainty-imputation[xgboost]'    # + XGBoost model
pip install 'uncertainty-imputation[dev]'        # + pytest, ruff
```

From source:

```bash
git clone https://github.com/rrwabina/uncertainty-imputation.git
cd uncertainty-imputation
pip install -e .
```

Requires Python 3.9+, `numpy`, `pandas`, `scipy`, `scikit-learn`. XGBoost is optional.

---

## Quickstart

```python
import numpy as np
import pandas as pd
from uncertainty_imputation import UncertaintyImputer, MCARAmputer
from uncertainty_imputation.utils.metrics import evaluate_imputation

# 1. Simulate missingness on a complete dataset
X_true = pd.DataFrame(np.random.default_rng(0).normal(size=(300, 4)),
                      columns=['a', 'b', 'c', 'd'])
X_missing = MCARAmputer(missing_rate=0.25, random_state=0).ampute(X_true)
mask = X_missing.isna().to_numpy()

# 2. Impute with a random forest + Expected Improvement acquisition
imputer = UncertaintyImputer(
    model='random_forest',
    acquisition='ei',
    max_iter=10,
    random_state=42,
)
X_imputed = imputer.fit_transform(X_missing)

# 3. Evaluate
print(evaluate_imputation(X_true, X_imputed, mask))
# {'rmse': 0.61, 'mae': 0.44, 'nrmse': 0.80, 'n_imputed': 285}

# 4. Inspect per-cell acquisition scores
for col_idx, scores in imputer.uncertainty_.items():
    print(f"col {col_idx}: mean acq={scores.mean():.3f}")
```

See [`examples/`](examples/) for four runnable scripts.

---

## Core concepts

### Imputation

`UncertaintyImputer` iteratively refines missing values. Each iteration, for every column with missingness:

1. Rows with the column observed become training data; rows missing become prediction targets.
2. An uncertainty-aware model predicts the missing values **plus a predictive standard deviation**.
3. An acquisition function turns `(mean, std)` into a per-prediction score, recorded in `imputer.uncertainty_`.
4. Missing cells are filled with the predicted means.

Iterates until `max_iter` is reached or the per-column normalised mean change drops below `tol`.

### Supported models

Each model exposes `predict_with_uncertainty(X) -> (mean, std)`.

| alias | class | uncertainty source |
|---|---|---|
| `'linear'` | `LinearRegressionModel` | bootstrap across `n_bootstrap` fits |
| `'decision_tree'` | `DecisionTreeModel` | bootstrap across `n_bootstrap` trees |
| `'random_forest'` | `RandomForestModel` | per-tree variance in the forest |
| `'xgboost'` | `XGBoostModel` | bootstrap across `n_bootstrap` XGB fits |

### Supported acquisition functions

All are vectorised and share the signature `acq(mean, std, y_best=None) -> scores`.

| alias | class | formula |
|---|---|---|
| `'us'` | `UncertaintySampling` | `std` |
| `'pi'` | `ProbabilityOfImprovement` | `Φ((y* − μ − ξ) / σ)` |
| `'ei'` | `ExpectedImprovement` | `(y* − μ − ξ)·Φ(z) + σ·φ(z)` |

### Amputation

Simulate missingness on complete datasets for controlled experiments:

```python
from uncertainty_imputation import MCARAmputer, MARAmputer, MNARAmputer

# Independent random missingness
X_mcar = MCARAmputer(missing_rate=0.25, random_state=0).ampute(X)

# Missingness in specific columns depends on other columns
X_mar = MARAmputer(missing_rate=0.25, columns=['y'],
                   predictor_columns=['x'], strength=4.0,
                   random_state=0).ampute(X)

# Missingness depends on the value itself (upper tail hidden)
X_mnar = MNARAmputer(missing_rate=0.25, direction='upper',
                     strength=1.0, random_state=0).ampute(X)
```

Every amputer accepts `return_mask=True` to also get the boolean mask, and all are seeded for reproducibility.

---

## API reference

### `UncertaintyImputer`

```python
UncertaintyImputer(
    model='random_forest',        # str alias or UncertaintyModel instance
    acquisition='us',             # str alias or AcquisitionFunction instance
    max_iter=10,
    tol=1e-3,                     # stop early if per-column mean abs change / col std < tol
    initial_strategy='mean',      # 'mean' | 'median'
    column_order='ascending',     # 'ascending' | 'descending' | 'random'
    min_obs_for_model=10,         # columns with fewer observed rows get init-filled
    update_rule='acquisition_weighted',  # 'mean' | 'acquisition_weighted' | 'sample'
    predictor_columns=None,       # dict: target -> list of predictor columns (see below)
    random_state=None,
    verbose=False,
    model_kwargs=None,            # dict forwarded to the model constructor
    acquisition_kwargs=None,      # dict forwarded to the acquisition constructor
)
```

**Restricting predictors per target.** By default, every column is imputed using every other column (the fully-conditional MICE specification). To encode domain knowledge — "BMI is a function of height and weight; blood pressure depends on age and BMI" — pass a `predictor_columns` dict:

```python
UncertaintyImputer(
    predictor_columns={
        'bmi':    ['height', 'weight'],
        'sbp':    ['age', 'bmi'],
        # columns not in the dict fall back to 'all other columns'
    },
)
```

Keys and values can be column names (for DataFrames) or integer indices (for either). Self-references are silently dropped. See [`examples/05_predictor_columns.py`](examples/05_predictor_columns.py) for a runnable demo.

Fitted attributes:

- `n_iter_` — iterations actually run.
- `history_` — list of per-iteration diagnostic dicts (change, per-column mean/std/acquisition).
- `uncertainty_` — dict mapping column index to per-missing-cell acquisition score array.

### Extending

Register custom components and reference them by string alias:

```python
from uncertainty_imputation import (
    AcquisitionFunction, UncertaintyModel,
    register_acquisition, register_model,
)

class MyAcquisition(AcquisitionFunction):
    name = 'mine'
    def score(self, mean, std, y_best=None):
        return ...  # 1D array

register_acquisition('mine', MyAcquisition)
UncertaintyImputer(acquisition='mine')  # works
```

Same pattern for models — subclass `UncertaintyModel`, implement `fit`, `predict`, `predict_with_uncertainty`, register.

---

## Benchmarks

Run `python examples/02_benchmark.py` to reproduce. On a 500×5 dataset with 25% MCAR missingness and nonlinear correlations, using the default `update_rule='acquisition_weighted'`:

| imputer               | model         | acquisition | RMSE      | MAE       | time (s) |
| --------------------- | ------------- | ----------- | --------- | --------- | -------- |
| UncertaintyImputer    | random_forest | ei          | **0.056** | **0.041** | 1.29     |
| UncertaintyImputer    | xgboost       | ei          | 0.061     | 0.045     | 13.1     |
| UncertaintyImputer    | linear        | ei          | 0.072     | 0.053     | 0.15     |
| UncertaintyImputer    | random_forest | pi          | 0.079     | 0.058     | 1.25     |
| UncertaintyImputer    | xgboost       | pi          | 0.084     | 0.061     | 12.9     |
| UncertaintyImputer    | linear        | pi          | 0.095     | 0.069     | 0.11     |
| UncertaintyImputer    | random_forest | us          | 0.108     | 0.078     | 1.22     |
| UncertaintyImputer    | decision_tree | us          | 0.116     | 0.084     | 0.45     |
| UncertaintyImputer    | linear        | us          | 0.129     | 0.093     | 0.17     |
| MICEImputer (sklearn) | —             | —           | 0.55      | 0.40      | 0.03     |
| MeanImputer           | —             | —           | 0.82      | 0.61      | 0.00     |


Every `UncertaintyImputer` configuration beats MICE and mean imputation. Expected Improvement is the strongest acquisition across models on this dataset - high-uncertainty cells get the largest updates, which aggressively corrects the cells where the model is most informative. The linear + EI combination is competitive with scikit-learn's MICE.

---

## Design notes

**Update rule.** The `update_rule` parameter controls how the acquisition score affects the imputed value:

- `'acquisition_weighted'` (default) — imputed value is `previous + α·(mean − previous)`, where `α ∈ [0, 1]` is the min-max-normalised acquisition score within the column. High-acquisition cells move aggressively toward the model's prediction; low-acquisition cells stay closer to their previous value. Different acquisition functions therefore produce genuinely different imputations.
- `'mean'` — imputed value is always the predictive mean. The acquisition score is recorded in `uncertainty_` but does not alter the update. All three acquisitions produce identical imputations; only the recorded scores differ. Use this when you want acquisition purely as an audit signal.
- `'sample'` — imputed value is drawn from `N(mean, std)`, giving classical "proper multiple imputation" semantics.

**Convergence.** The tolerance is on the per-column mean absolute change between iterations divided by the column's current standard deviation (matching sklearn's `IterativeImputer` convention). This is scale-invariant and robust to features with values near zero.

**Caching.** Calling `transform(X)` on the same matrix passed to `fit(X)` returns the cached result. Calling it on a different matrix re-runs iteration using the stored initialisation values.

**Reproducibility.** Every stochastic component accepts `random_state`. Two runs with the same seed and identical inputs produce bit-identical outputs (verified by tests).

---

## Testing

```bash
pip install -e '.[dev,xgboost]'
pytest tests/                 # 111 tests
pytest --cov=uncertainty_imputation tests/
```

---

## Project layout

```
uncertainty_imputation/
├── src/uncertainty_imputation/
│   ├── imputation/
│   │   ├── base.py              # BaseImputer
│   │   ├── iterative.py         # UncertaintyImputer
│   │   ├── models.py            # linear, tree, RF, XGBoost wrappers + registry
│   │   ├── acquisition.py       # US, PI, EI + registry
│   │   └── baselines.py         # MeanImputer, MICEImputer
│   ├── amputation/
│   │   ├── base.py              # BaseAmputer
│   │   ├── mcar.py, mar.py, mnar.py
│   └── utils/
│       ├── metrics.py           # rmse, mae, nrmse, per-column RMSE
│       └── preprocessing.py     # missing masks, standardisation
├── tests/                       # 99 unit tests
├── examples/
│   ├── 01_quickstart.py
│   ├── 02_benchmark.py
│   ├── 03_amputation.py
│   └── 04_custom_components.py
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If this library supports your research, please cite the original paper:

```bibtex
@article{wabina2025uncertainty,
  title={Uncertainty-aware approach for multiple imputation using conventional and machine learning models: a real-world data study},
  author={Wabina, Romen Samuel and Looareesuwan, Panu and Sonsilphong, Suphachoke and Teza, Htun and Ponthongmak, Wanchana and McKay, Gareth and Attia, John and Pattanateepapon, Anuchate and Panitchote, Anupol and Thakkinstian, Ammarin},
  journal={Journal of Big Data},
  volume={12},
  number={1},
  pages={95},
  year={2025},
  publisher={Springer}
}
```
