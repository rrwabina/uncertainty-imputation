"""Tests for imputers: UncertaintyImputer, MeanImputer, MICEImputer."""

import numpy as np
import pandas as pd
import pytest

from uncertainty_imputation import (
    MCARAmputer,
    MeanImputer,
    MICEImputer,
    UncertaintyImputer,
)
from uncertainty_imputation.utils.metrics import rmse


def _make_dataset(n=150, d=4, missing_rate=0.2, seed=0):
    '''Build a simple correlated dataset with MCAR missingness.'''
    rng = np.random.default_rng(seed)
    # Correlated features: X1 drives X2, X3; X4 is independent
    X1 = rng.normal(size=n)
    X2 = X1 + 0.2 * rng.normal(size=n)
    X3 = -X1 + 0.2 * rng.normal(size=n)
    X4 = rng.normal(size=n)
    X_true = np.stack([X1, X2, X3, X4], axis=1)
    X_missing = MCARAmputer(
        missing_rate=missing_rate, random_state=seed
    ).ampute(X_true)
    return X_true, np.asarray(X_missing)


class TestUncertaintyImputerBasic:
    def test_shape_preserved(self):
        X_true, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', acquisition='us', max_iter=3, random_state=0
        )
        X_out = imp.fit_transform(X_missing)
        assert X_out.shape == X_missing.shape

    def test_no_nans_remain(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', acquisition='us', max_iter=3, random_state=0
        )
        X_out = imp.fit_transform(X_missing)
        assert not np.isnan(np.asarray(X_out)).any()

    def test_observed_values_preserved(self):
        _, X_missing = _make_dataset()
        mask = np.isnan(X_missing)
        imp = UncertaintyImputer(
            model='linear', acquisition='us', max_iter=3, random_state=0
        )
        X_out = np.asarray(imp.fit_transform(X_missing))
        # Observed entries should be unchanged
        np.testing.assert_allclose(X_out[~mask], X_missing[~mask])

    def test_beats_mean_imputer(self):
        # A non-trivial end-to-end sanity check: with correlated features,
        # the model-based imputer should beat mean imputation.
        X_true, X_missing = _make_dataset(n=300, missing_rate=0.2, seed=1)
        mask = np.isnan(X_missing)

        mean_imp = MeanImputer().fit_transform(X_missing)
        rf_imp = UncertaintyImputer(
            model='random_forest',
            acquisition='us',
            max_iter=5,
            random_state=0,
            model_kwargs={'n_estimators': 30},
        ).fit_transform(X_missing)

        rmse_mean = rmse(X_true, mean_imp, mask)
        rmse_rf = rmse(X_true, rf_imp, mask)
        assert rmse_rf < rmse_mean, (
            f'Expected RF imputer to beat mean imputer; '
            f'got RMSE {rmse_rf:.4f} vs {rmse_mean:.4f}'
        )

    def test_reproducibility(self):
        _, X_missing = _make_dataset()
        imp1 = UncertaintyImputer(
            model='random_forest', acquisition='ei', max_iter=3,
            random_state=42, model_kwargs={'n_estimators': 10},
        ).fit_transform(X_missing)
        imp2 = UncertaintyImputer(
            model='random_forest', acquisition='ei', max_iter=3,
            random_state=42, model_kwargs={'n_estimators': 10},
        ).fit_transform(X_missing)
        np.testing.assert_allclose(np.asarray(imp1), np.asarray(imp2))

    def test_tracks_uncertainty(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', acquisition='us', max_iter=2, random_state=0
        )
        imp.fit(X_missing)
        mask = np.isnan(X_missing)
        # Every column with missing values should have a score vector
        for j in range(X_missing.shape[1]):
            n_missing = int(mask[:, j].sum())
            if n_missing > 0:
                assert j in imp.uncertainty_
                assert imp.uncertainty_[j].shape == (n_missing,)

    def test_history_recorded(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', acquisition='us', max_iter=3, random_state=0
        ).fit(X_missing)
        assert len(imp.history_) == imp.n_iter_
        assert all('change' in h for h in imp.history_)


class TestUncertaintyImputerDataFrame:
    def test_dataframe_roundtrip(self):
        X_true, X_missing_arr = _make_dataset()
        X_missing = pd.DataFrame(
            X_missing_arr, columns=['a', 'b', 'c', 'd']
        )
        imp = UncertaintyImputer(
            model='linear', max_iter=2, random_state=0
        ).fit_transform(X_missing)
        assert isinstance(imp, pd.DataFrame)
        assert list(imp.columns) == ['a', 'b', 'c', 'd']
        assert (imp.index == X_missing.index).all()

    def test_numpy_returns_numpy(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=2, random_state=0
        ).fit_transform(X_missing)
        assert isinstance(imp, np.ndarray)


class TestUncertaintyImputerConfig:
    def test_all_acquisition_functions_run(self):
        _, X_missing = _make_dataset()
        for acq in ['us', 'pi', 'ei']:
            imp = UncertaintyImputer(
                model='linear',
                acquisition=acq,
                max_iter=2,
                random_state=0,
            ).fit_transform(X_missing)
            assert not np.isnan(np.asarray(imp)).any(), f'{acq} left NaNs'

    def test_all_models_run(self):
        _, X_missing = _make_dataset(n=80)
        for model in ['linear', 'decision_tree', 'random_forest']:
            imp = UncertaintyImputer(
                model=model,
                acquisition='us',
                max_iter=2,
                random_state=0,
                model_kwargs={'n_estimators': 10} if model == 'random_forest' else {},
            ).fit_transform(X_missing)
            assert not np.isnan(np.asarray(imp)).any(), f'{model} left NaNs'

    def test_column_order_options(self):
        _, X_missing = _make_dataset()
        for order in ['ascending', 'descending', 'random']:
            imp = UncertaintyImputer(
                model='linear', max_iter=2, column_order=order, random_state=0,
            ).fit_transform(X_missing)
            assert not np.isnan(np.asarray(imp)).any()

    def test_invalid_column_order(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=1, column_order='backwards', random_state=0
        )
        with pytest.raises(ValueError, match='column_order'):
            imp.fit(X_missing)

    def test_invalid_initial_strategy(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=1, initial_strategy='mode', random_state=0
        )
        with pytest.raises(ValueError, match='initial_strategy'):
            imp.fit(X_missing)

    def test_convergence_early_stop(self):
        # With high tolerance, should stop after 1 iteration
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=10, tol=1e10, random_state=0
        ).fit(X_missing)
        assert imp.n_iter_ == 1


class TestUncertaintyImputerTransform:
    def test_transform_cached_on_fit_data(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=2, random_state=0
        ).fit(X_missing)
        out = imp.transform(X_missing)
        assert not np.isnan(np.asarray(out)).any()

    def test_transform_new_data(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=2, random_state=0
        ).fit(X_missing)
        # Different missingness pattern
        _, X_other = _make_dataset(seed=99)
        out = imp.transform(X_other)
        assert out.shape == X_other.shape
        assert not np.isnan(np.asarray(out)).any()

    def test_transform_shape_mismatch_raises(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(model='linear', max_iter=1).fit(X_missing)
        with pytest.raises(ValueError, match='features'):
            imp.transform(np.zeros((5, 99)))

    def test_transform_before_fit_raises(self):
        imp = UncertaintyImputer()
        with pytest.raises(RuntimeError, match='fitted'):
            imp.transform(np.zeros((5, 3)))


class TestMeanImputer:
    def test_basic(self):
        X_true, X_missing = _make_dataset()
        mask = np.isnan(X_missing)
        out = MeanImputer().fit_transform(X_missing)
        assert not np.isnan(np.asarray(out)).any()
        # Imputed values should all be close to their column means
        means = np.nanmean(X_missing, axis=0)
        for j in range(X_missing.shape[1]):
            if mask[:, j].any():
                np.testing.assert_allclose(out[mask[:, j], j], means[j])

    def test_median_strategy(self):
        _, X_missing = _make_dataset()
        out = MeanImputer(strategy='median').fit_transform(X_missing)
        assert not np.isnan(np.asarray(out)).any()

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            MeanImputer(strategy='mode')


class TestMICEImputer:
    def test_basic(self):
        X_true, X_missing = _make_dataset(n=80)
        out = MICEImputer(max_iter=3, random_state=0).fit_transform(X_missing)
        assert not np.isnan(np.asarray(out)).any()
        assert out.shape == X_missing.shape


class TestUpdateRule:
    '''The update_rule parameter controls how acquisition affects the fill.'''

    def test_mean_rule_acquisition_agnostic(self):
        # With update_rule='mean', all three acquisitions produce identical
        # imputed values (the rule ignores the score).
        _, X_missing = _make_dataset()
        out_us = UncertaintyImputer(
            model='linear', acquisition='us', update_rule='mean',
            max_iter=3, random_state=0,
        ).fit_transform(X_missing)
        out_ei = UncertaintyImputer(
            model='linear', acquisition='ei', update_rule='mean',
            max_iter=3, random_state=0,
        ).fit_transform(X_missing)
        np.testing.assert_allclose(np.asarray(out_us), np.asarray(out_ei))

    def test_acquisition_weighted_differs_from_mean(self):
        # With update_rule='acquisition_weighted', different acquisitions
        # produce different imputations (as the weights differ per-cell).
        _, X_missing = _make_dataset()
        out_us = np.asarray(UncertaintyImputer(
            model='random_forest', acquisition='us',
            update_rule='acquisition_weighted', max_iter=3, random_state=0,
            model_kwargs={'n_estimators': 15},
        ).fit_transform(X_missing))
        out_mean = np.asarray(UncertaintyImputer(
            model='random_forest', acquisition='us',
            update_rule='mean', max_iter=3, random_state=0,
            model_kwargs={'n_estimators': 15},
        ).fit_transform(X_missing))
        # Not identical
        assert not np.allclose(out_us, out_mean)

    def test_sample_rule_stochastic_respects_seed(self):
        # With update_rule='sample', values are drawn from N(mean, std).
        # Same seed -> same outputs.
        _, X_missing = _make_dataset()
        out1 = np.asarray(UncertaintyImputer(
            model='random_forest', update_rule='sample',
            max_iter=2, random_state=42,
            model_kwargs={'n_estimators': 10},
        ).fit_transform(X_missing))
        out2 = np.asarray(UncertaintyImputer(
            model='random_forest', update_rule='sample',
            max_iter=2, random_state=42,
            model_kwargs={'n_estimators': 10},
        ).fit_transform(X_missing))
        np.testing.assert_allclose(out1, out2)
        assert not np.isnan(out1).any()

    def test_invalid_update_rule_raises(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', update_rule='nonsense', max_iter=1, random_state=0
        )
        with pytest.raises(ValueError, match='update_rule'):
            imp.fit(X_missing)


class TestPredictorColumns:
    '''The predictor_columns parameter restricts the predictor set per target.'''

    def test_default_is_all_other_columns(self):
        # With predictor_columns=None, _predictors_for should return all
        # other columns (the pre-existing behavior).
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=1, random_state=0
        ).fit(X_missing)
        preds = imp._predictors_for(2, X_missing.shape[1])
        assert preds == [0, 1, 3]

    def test_numpy_int_keys(self):
        # Target index 1 should use only column 0 as predictor
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear',
            max_iter=2,
            random_state=0,
            predictor_columns={1: [0]},  # impute col 1 using only col 0
        ).fit(X_missing)
        # Target 1 has a restricted predictor list
        assert imp._predictors_for(1, 4) == [0]
        # Other targets fall back to all-others
        assert imp._predictors_for(0, 4) == [1, 2, 3]
        assert imp._predictors_for(2, 4) == [0, 1, 3]

    def test_dataframe_name_keys(self):
        X_true, X_arr = _make_dataset()
        X = pd.DataFrame(X_arr, columns=['a', 'b', 'c', 'd'])
        imp = UncertaintyImputer(
            model='linear',
            max_iter=2,
            random_state=0,
            predictor_columns={'b': ['a', 'c'], 'd': ['a']},
        )
        imp.fit(X)
        # Names resolved to indices
        assert imp._predictors_for(1, 4) == [0, 2]  # 'b' -> ['a', 'c']
        assert imp._predictors_for(3, 4) == [0]     # 'd' -> ['a']

    def test_mixed_int_and_name_keys(self):
        # Users can mix int and str keys against a DataFrame
        X_true, X_arr = _make_dataset()
        X = pd.DataFrame(X_arr, columns=['a', 'b', 'c', 'd'])
        imp = UncertaintyImputer(
            model='linear',
            max_iter=1,
            random_state=0,
            predictor_columns={'b': [0, 'c'], 2: ['a']},
        ).fit(X)
        assert imp._predictors_for(1, 4) == [0, 2]
        assert imp._predictors_for(2, 4) == [0]

    def test_self_reference_silently_dropped(self):
        # A target listed as its own predictor must be filtered out
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=1, random_state=0,
            predictor_columns={0: [0, 1, 2]},
        ).fit(X_missing)
        assert imp._predictors_for(0, 4) == [1, 2]

    def test_duplicate_predictors_deduplicated(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=1, random_state=0,
            predictor_columns={0: [1, 2, 1, 2, 3]},
        ).fit(X_missing)
        assert imp._predictors_for(0, 4) == [1, 2, 3]

    def test_empty_predictor_set_falls_back_to_init(self):
        # If the resolved predictor list is empty, the target is filled
        # with the initialisation value (no model is fit).
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear',
            max_iter=2,
            random_state=0,
            initial_strategy='mean',
            predictor_columns={0: [0]},  # only self -> empty after filter
        )
        X_out = np.asarray(imp.fit_transform(X_missing))
        mask = np.isnan(X_missing)
        # All imputed cells in column 0 should equal the initial mean
        init_mean = float(np.nanmean(X_missing[:, 0]))
        np.testing.assert_allclose(
            X_out[mask[:, 0], 0], init_mean, atol=1e-8
        )
        # No NaNs remain overall
        assert not np.isnan(X_out).any()

    def test_restricted_predictors_affect_imputation(self):
        # Using a tiny predictor set should produce a different result
        # than using all predictors.
        _, X_missing = _make_dataset()
        out_all = np.asarray(UncertaintyImputer(
            model='linear', max_iter=3, random_state=0, update_rule='mean',
        ).fit_transform(X_missing))
        out_restricted = np.asarray(UncertaintyImputer(
            model='linear', max_iter=3, random_state=0, update_rule='mean',
            predictor_columns={0: [1], 1: [0], 2: [3], 3: [2]},
        ).fit_transform(X_missing))
        mask = np.isnan(X_missing)
        # The two imputations should genuinely differ on the imputed cells
        assert not np.allclose(out_all[mask], out_restricted[mask])

    def test_unknown_name_raises(self):
        X_true, X_arr = _make_dataset()
        X = pd.DataFrame(X_arr, columns=['a', 'b', 'c', 'd'])
        imp = UncertaintyImputer(
            model='linear', max_iter=1, random_state=0,
            predictor_columns={'b': ['nonexistent']},
        )
        with pytest.raises(KeyError, match='nonexistent'):
            imp.fit(X)

    def test_out_of_range_int_raises(self):
        _, X_missing = _make_dataset()  # 4-feature matrix
        imp = UncertaintyImputer(
            model='linear', max_iter=1, random_state=0,
            predictor_columns={0: [99]},
        )
        with pytest.raises(ValueError, match='out of range'):
            imp.fit(X_missing)

    def test_string_name_with_numpy_raises(self):
        _, X_missing = _make_dataset()  # numpy input
        imp = UncertaintyImputer(
            model='linear', max_iter=1, random_state=0,
            predictor_columns={'a': ['b']},
        )
        with pytest.raises(TypeError, match='pandas DataFrame'):
            imp.fit(X_missing)

    def test_bad_key_type_raises(self):
        _, X_missing = _make_dataset()
        imp = UncertaintyImputer(
            model='linear', max_iter=1, random_state=0,
            predictor_columns={0: [1.5]},  # float is invalid
        )
        with pytest.raises(TypeError):
            imp.fit(X_missing)


try:
    import xgboost  # noqa: F401
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


@pytest.mark.skipif(not _HAS_XGB, reason='xgboost not installed')
class TestUncertaintyImputerXGBoostEI:
    '''End-to-end test of the config highlighted in the README/docstring.'''

    def test_xgboost_ei_runs_and_imputes(self):
        X_true, X_missing = _make_dataset(n=120)
        imp = UncertaintyImputer(
            model='xgboost',
            acquisition='ei',
            max_iter=3,
            random_state=42,
            model_kwargs={'n_bootstrap': 3, 'n_estimators': 30},
        )
        X_out = np.asarray(imp.fit_transform(X_missing))
        assert not np.isnan(X_out).any()
        # Uncertainty scores recorded for columns with missingness
        mask = np.isnan(X_missing)
        for j in range(X_missing.shape[1]):
            if mask[:, j].any():
                assert j in imp.uncertainty_
