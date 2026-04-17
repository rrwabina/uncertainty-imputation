"""Tests for utility functions: metrics and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from uncertainty_imputation.utils.metrics import (
    evaluate_imputation,
    mae,
    normalized_rmse,
    per_column_rmse,
    rmse,
)
from uncertainty_imputation.utils.preprocessing import (
    get_missing_mask,
    missing_summary,
    standardize,
    unstandardize,
)


class TestMetrics:
    def test_rmse_basic(self):
        X_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_imp = np.array([[1.1, 2.0], [3.0, 3.8]])
        mask = np.array([[True, False], [False, True]])
        # Errors: 0.1 and 0.2 -> RMSE = sqrt((0.01 + 0.04)/2)
        expected = float(np.sqrt((0.01 + 0.04) / 2))
        assert abs(rmse(X_true, X_imp, mask) - expected) < 1e-10

    def test_mae_basic(self):
        X_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_imp = np.array([[1.1, 2.0], [3.0, 3.8]])
        mask = np.array([[True, False], [False, True]])
        expected = (0.1 + 0.2) / 2
        assert abs(mae(X_true, X_imp, mask) - expected) < 1e-10

    def test_metrics_only_on_masked_cells(self):
        # Changing an unmasked cell should not affect the metric
        X_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_imp1 = X_true.copy()
        X_imp1[0, 0] = 99.0  # huge error, but unmasked
        X_imp1[1, 1] = 4.1   # small error, masked
        mask = np.array([[False, False], [False, True]])
        assert rmse(X_true, X_imp1, mask) == pytest.approx(0.1)

    def test_normalized_rmse(self):
        X_true = np.array([[1.0, 2.0, 3.0, 4.0]])
        X_imp = X_true.copy()
        mask = np.ones_like(X_true, dtype=bool)
        # Perfect imputation -> 0
        assert normalized_rmse(X_true, X_imp, mask) == 0.0

    def test_per_column_rmse(self):
        X_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_imp = np.array([[1.1, 2.0], [3.0, 4.2], [5.3, 6.0]])
        mask = np.array([[True, False], [False, True], [True, False]])
        out = per_column_rmse(X_true, X_imp, mask)
        # Col 0: errors 0.1, 0.3 -> RMSE sqrt((0.01+0.09)/2)
        # Col 1: error 0.2 -> RMSE 0.2
        assert abs(out[0] - np.sqrt((0.01 + 0.09) / 2)) < 1e-10
        assert abs(out[1] - 0.2) < 1e-10

    def test_per_column_rmse_empty_column(self):
        X_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_imp = np.array([[1.0, 2.1], [3.0, 4.0]])
        mask = np.array([[False, True], [False, False]])
        out = per_column_rmse(X_true, X_imp, mask)
        assert np.isnan(out[0])  # no imputed cells in col 0
        assert abs(out[1] - 0.1) < 1e-10

    def test_evaluate_imputation_dict(self):
        X_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_imp = np.array([[1.1, 2.0], [3.0, 3.9]])
        mask = np.array([[True, False], [False, True]])
        result = evaluate_imputation(X_true, X_imp, mask)
        assert 'rmse' in result
        assert 'mae' in result
        assert 'nrmse' in result
        assert result['n_imputed'] == 2

    def test_shape_mismatch(self):
        X_true = np.zeros((2, 3))
        X_imp = np.zeros((2, 4))
        mask = np.zeros((2, 3), dtype=bool)
        with pytest.raises(ValueError, match='Shape mismatch'):
            rmse(X_true, X_imp, mask)

    def test_empty_mask(self):
        X_true = np.zeros((2, 3))
        X_imp = np.zeros((2, 3))
        mask = np.zeros((2, 3), dtype=bool)
        with pytest.raises(ValueError, match='no missing cells'):
            rmse(X_true, X_imp, mask)

    def test_dataframe_input(self):
        X_true = pd.DataFrame({'a': [1.0, 3.0], 'b': [2.0, 4.0]})
        X_imp = pd.DataFrame({'a': [1.1, 3.0], 'b': [2.0, 3.9]})
        mask = np.array([[True, False], [False, True]])
        result = rmse(X_true, X_imp, mask)
        expected = float(np.sqrt((0.01 + 0.01) / 2))
        assert abs(result - expected) < 1e-10


class TestPreprocessing:
    def test_get_missing_mask_numpy(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0]])
        mask = get_missing_mask(X)
        np.testing.assert_array_equal(mask, [[False, True], [False, False]])

    def test_get_missing_mask_dataframe(self):
        df = pd.DataFrame({'a': [1.0, np.nan], 'b': [2.0, 3.0]})
        mask = get_missing_mask(df)
        np.testing.assert_array_equal(mask, [[False, False], [True, False]])

    def test_missing_summary_numpy(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, np.nan]])
        summary = missing_summary(X)
        assert summary.iloc[0]['n_missing'] == 1
        assert summary.iloc[1]['n_missing'] == 2
        assert abs(summary.iloc[0]['missing_rate'] - 1/3) < 1e-10

    def test_missing_summary_dataframe(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [1.0, 2.0, 3.0]})
        summary = missing_summary(df)
        assert summary.loc['a', 'n_missing'] == 1
        assert summary.loc['b', 'n_missing'] == 0

    def test_standardize_numpy(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        out = standardize(X)
        np.testing.assert_allclose(np.mean(out, axis=0), [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(np.std(out, axis=0), [1.0, 1.0], atol=1e-10)

    def test_standardize_with_nans(self):
        X = np.array([[1.0, np.nan], [2.0, 20.0], [3.0, 30.0]])
        out = standardize(X)
        # Non-NaN cells finite
        assert np.all(np.isfinite(out[~np.isnan(X)]))

    def test_standardize_roundtrip(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        out, mean, std = standardize(X, return_stats=True)
        recovered = unstandardize(out, mean, std)
        np.testing.assert_allclose(recovered, X)

    def test_standardize_dataframe(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [10.0, 20.0, 30.0]})
        out = standardize(df)
        assert isinstance(out, pd.DataFrame)
        np.testing.assert_allclose(out.mean().values, [0.0, 0.0], atol=1e-10)

    def test_standardize_zero_variance(self):
        # Constant column shouldn't produce inf/nan
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        out = standardize(X)
        assert np.all(np.isfinite(out))
