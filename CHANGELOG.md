# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-04-17

### Added

- `predictor_columns` parameter on `UncertaintyImputer`. Restrict which
  columns are used as predictors when imputing each target column, matching
  the `predictor matrix` concept from R's `mice` package. Accepts a dict
  mapping target → list of predictors, using either string names (for
  DataFrame input) or integer indices (for either). Targets not in the
  dict fall back to the default "all other columns" behavior. Self-
  references are silently filtered; duplicates are deduplicated.
- `examples/05_predictor_columns.py` demonstrating the feature on a
  synthetic clinical dataset.

### Changed

- Test count: 99 → 111 (12 new tests for `predictor_columns`).

## [0.1.0] — 2026-04-17

### Added

- Initial release.
- `UncertaintyImputer`: iterative, uncertainty-aware multiple imputation
  with configurable model, acquisition function, and update rule.
- Models: `LinearRegressionModel`, `DecisionTreeModel`, `RandomForestModel`,
  `XGBoostModel` (optional dependency), all exposing a uniform
  `predict_with_uncertainty(X) -> (mean, std)` interface.
- Acquisition functions: `UncertaintySampling`, `ProbabilityOfImprovement`,
  `ExpectedImprovement`, plus a registry for custom subclasses.
- Update rules: `'mean'`, `'acquisition_weighted'` (default), `'sample'`
  controlling how `(mean, std, acquisition_score)` combine into the
  imputed value.
- Amputation module: `MCARAmputer`, `MARAmputer`, `MNARAmputer` for
  generating controlled missingness on complete datasets.
- Baseline imputers for benchmarking: `MeanImputer`, `MICEImputer` (thin
  wrapper around sklearn's `IterativeImputer`).
- Utilities: metrics (`rmse`, `mae`, `normalized_rmse`, `per_column_rmse`,
  `evaluate_imputation`) and preprocessing (`get_missing_mask`,
  `missing_summary`, `standardize`, `unstandardize`).
- Registries for custom models and acquisition functions via
  `register_model` and `register_acquisition`.
- 99 unit tests, four runnable example scripts, `pyproject.toml` ready for
  PyPI upload, MIT license.
