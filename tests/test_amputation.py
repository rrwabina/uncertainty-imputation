"""Tests for amputation (missing data simulation)."""

import numpy as np
import pandas as pd
import pytest

from uncertainty_imputation import MARAmputer, MCARAmputer, MNARAmputer


def _complete_data(n=500, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


class TestMCARAmputer:
    def test_shape_preserved(self):
        X = _complete_data()
        amp = MCARAmputer(missing_rate=0.2, random_state=0)
        X_out = amp.ampute(X)
        assert X_out.shape == X.shape

    def test_rate_approximately_correct(self):
        X = _complete_data(n=2000, d=5)
        amp = MCARAmputer(missing_rate=0.3, random_state=0)
        X_out = np.asarray(amp.ampute(X))
        observed_rate = np.isnan(X_out).mean()
        assert abs(observed_rate - 0.3) < 0.02, observed_rate

    def test_zero_rate(self):
        X = _complete_data()
        X_out = np.asarray(MCARAmputer(missing_rate=0.0).ampute(X))
        assert not np.isnan(X_out).any()

    def test_reproducible(self):
        X = _complete_data()
        X1 = np.asarray(MCARAmputer(missing_rate=0.2, random_state=42).ampute(X))
        X2 = np.asarray(MCARAmputer(missing_rate=0.2, random_state=42).ampute(X))
        np.testing.assert_array_equal(np.isnan(X1), np.isnan(X2))

    def test_return_mask(self):
        X = _complete_data()
        amp = MCARAmputer(missing_rate=0.2, random_state=0)
        X_out, mask = amp.ampute(X, return_mask=True)
        assert mask.dtype == bool
        assert mask.shape == X.shape
        # Mask should agree with isnan on the output
        np.testing.assert_array_equal(mask, np.isnan(np.asarray(X_out)))

    def test_subset_columns(self):
        X = _complete_data(d=5)
        amp = MCARAmputer(missing_rate=0.5, columns=[0, 2], random_state=0)
        X_out = np.asarray(amp.ampute(X))
        # Only columns 0 and 2 should have any NaNs
        assert np.isnan(X_out[:, 0]).any()
        assert np.isnan(X_out[:, 2]).any()
        assert not np.isnan(X_out[:, 1]).any()
        assert not np.isnan(X_out[:, 3]).any()
        assert not np.isnan(X_out[:, 4]).any()

    def test_dataframe_columns_by_name(self):
        X = pd.DataFrame(_complete_data(), columns=['a', 'b', 'c', 'd', 'e'])
        amp = MCARAmputer(missing_rate=0.5, columns=['a', 'c'], random_state=0)
        X_out = amp.ampute(X)
        assert isinstance(X_out, pd.DataFrame)
        assert X_out['a'].isna().any()
        assert X_out['c'].isna().any()
        assert not X_out['b'].isna().any()

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            MCARAmputer(missing_rate=1.5)
        with pytest.raises(ValueError):
            MCARAmputer(missing_rate=-0.1)

    def test_callable_interface(self):
        X = _complete_data()
        amp = MCARAmputer(missing_rate=0.2, random_state=0)
        X_out = amp(X)
        assert np.isnan(np.asarray(X_out)).any()


class TestMARAmputer:
    def test_shape_preserved(self):
        X = _complete_data()
        amp = MARAmputer(missing_rate=0.2, random_state=0)
        X_out = amp.ampute(X)
        assert X_out.shape == X.shape

    def test_rate_approximately_correct(self):
        X = _complete_data(n=2000, d=5)
        amp = MARAmputer(missing_rate=0.25, random_state=0)
        X_out = np.asarray(amp.ampute(X))
        # MAR quantile-thresholds to hit target rate exactly per column
        observed_rate = np.isnan(X_out).mean()
        assert abs(observed_rate - 0.25) < 0.02, observed_rate

    def test_missingness_depends_on_observed(self):
        # MAR means prob of missingness in col j depends on other cols.
        # Fit: generate X, ampute col 0 based on col 1. Then col 1 values
        # should differ between rows where col 0 is missing vs present.
        rng = np.random.default_rng(0)
        n = 2000
        x1 = rng.normal(size=n)
        x0 = 0.5 * x1 + rng.normal(size=n)
        X = np.stack([x0, x1], axis=1)

        amp = MARAmputer(
            missing_rate=0.3,
            columns=[0],
            predictor_columns=[1],
            strength=5.0,
            random_state=0,
        )
        X_out, mask = amp.ampute(X, return_mask=True)
        x1_missing = x1[mask[:, 0]]
        x1_present = x1[~mask[:, 0]]
        # Means should differ meaningfully (missingness depends on x1)
        assert abs(x1_missing.mean() - x1_present.mean()) > 0.5

    def test_reproducible(self):
        X = _complete_data()
        m1 = np.isnan(np.asarray(MARAmputer(missing_rate=0.2, random_state=7).ampute(X)))
        m2 = np.isnan(np.asarray(MARAmputer(missing_rate=0.2, random_state=7).ampute(X)))
        np.testing.assert_array_equal(m1, m2)


class TestMNARAmputer:
    def test_shape_preserved(self):
        X = _complete_data()
        X_out = MNARAmputer(missing_rate=0.2, random_state=0).ampute(X)
        assert X_out.shape == X.shape

    def test_upper_direction_hides_large_values(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1000, 2))
        amp = MNARAmputer(
            missing_rate=0.3, direction='upper', strength=1.0, random_state=0
        )
        X_out, mask = amp.ampute(X, return_mask=True)
        # For strength=1.0, the top 30% of column values are masked
        missing_mean = X[mask[:, 0], 0].mean()
        present_mean = X[~mask[:, 0], 0].mean()
        assert missing_mean > present_mean

    def test_lower_direction_hides_small_values(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1000, 2))
        amp = MNARAmputer(
            missing_rate=0.3, direction='lower', strength=1.0, random_state=0
        )
        X_out, mask = amp.ampute(X, return_mask=True)
        missing_mean = X[mask[:, 0], 0].mean()
        present_mean = X[~mask[:, 0], 0].mean()
        assert missing_mean < present_mean

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            MNARAmputer(direction='sideways')

    def test_invalid_strength(self):
        with pytest.raises(ValueError):
            MNARAmputer(strength=1.5)

    def test_rate_approximately_correct(self):
        X = _complete_data(n=2000)
        X_out = np.asarray(
            MNARAmputer(missing_rate=0.25, strength=1.0, random_state=0).ampute(X)
        )
        observed_rate = np.isnan(X_out).mean()
        assert abs(observed_rate - 0.25) < 0.02
