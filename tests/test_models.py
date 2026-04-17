"""Tests for uncertainty-aware model wrappers."""

import numpy as np
import pytest

from uncertainty_imputation.imputation.models import (
    DecisionTreeModel,
    LinearRegressionModel,
    RandomForestModel,
    UncertaintyModel,
    get_model,
    register_model,
)

try:
    from uncertainty_imputation.imputation.models import XGBoostModel, _HAS_XGBOOST
except ImportError:
    _HAS_XGBOOST = False


def _make_regression(n=200, d=4, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + noise * rng.normal(size=n)
    return X, y


class TestLinearRegressionModel:
    def test_fit_predict_shape(self):
        X, y = _make_regression()
        model = LinearRegressionModel(n_bootstrap=5, random_state=0)
        model.fit(X, y)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == (X.shape[0],)
        assert std.shape == (X.shape[0],)
        assert np.all(std >= 0)

    def test_point_predict_matches_bootstrap_mean(self):
        X, y = _make_regression()
        model = LinearRegressionModel(n_bootstrap=5, random_state=0).fit(X, y)
        point = model.predict(X)
        mean, _ = model.predict_with_uncertainty(X)
        np.testing.assert_allclose(point, mean)

    def test_reasonable_accuracy(self):
        X, y = _make_regression(n=500, noise=0.1, seed=1)
        model = LinearRegressionModel(n_bootstrap=10, random_state=0).fit(X, y)
        rmse = np.sqrt(np.mean((model.predict(X) - y) ** 2))
        # Should be on the order of the noise level, not dramatically worse
        assert rmse < 0.5

    def test_not_fitted_raises(self):
        model = LinearRegressionModel()
        with pytest.raises(RuntimeError, match='fitted first'):
            model.predict_with_uncertainty(np.zeros((1, 3)))

    def test_reproducible(self):
        X, y = _make_regression()
        m1 = LinearRegressionModel(n_bootstrap=5, random_state=42).fit(X, y)
        m2 = LinearRegressionModel(n_bootstrap=5, random_state=42).fit(X, y)
        mean1, std1 = m1.predict_with_uncertainty(X)
        mean2, std2 = m2.predict_with_uncertainty(X)
        np.testing.assert_allclose(mean1, mean2)
        np.testing.assert_allclose(std1, std2)


class TestDecisionTreeModel:
    def test_fit_predict_shape(self):
        X, y = _make_regression()
        model = DecisionTreeModel(n_bootstrap=5, random_state=0).fit(X, y)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == (X.shape[0],)
        assert std.shape == (X.shape[0],)
        assert np.all(std >= 0)

    def test_reproducible(self):
        X, y = _make_regression()
        m1 = DecisionTreeModel(n_bootstrap=5, random_state=42, max_depth=5).fit(X, y)
        m2 = DecisionTreeModel(n_bootstrap=5, random_state=42, max_depth=5).fit(X, y)
        mean1, _ = m1.predict_with_uncertainty(X)
        mean2, _ = m2.predict_with_uncertainty(X)
        np.testing.assert_allclose(mean1, mean2)


class TestRandomForestModel:
    def test_fit_predict_shape(self):
        X, y = _make_regression()
        model = RandomForestModel(n_estimators=20, random_state=0).fit(X, y)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == (X.shape[0],)
        assert std.shape == (X.shape[0],)
        assert np.all(std >= 0)

    def test_predict_matches_sklearn(self):
        # predict() should match the sklearn forest's point prediction,
        # and the mean across trees should equal the forest's predict().
        X, y = _make_regression()
        model = RandomForestModel(n_estimators=10, random_state=0).fit(X, y)
        mean, _ = model.predict_with_uncertainty(X)
        np.testing.assert_allclose(model.predict(X), mean)


@pytest.mark.skipif(not _HAS_XGBOOST, reason='xgboost not installed')
class TestXGBoostModel:
    def test_fit_predict_shape(self):
        X, y = _make_regression()
        model = XGBoostModel(n_bootstrap=3, random_state=0, n_estimators=20).fit(X, y)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == (X.shape[0],)
        assert std.shape == (X.shape[0],)
        assert np.all(std >= 0)


class TestGetModel:
    def test_string_aliases(self):
        assert isinstance(get_model('linear'), LinearRegressionModel)
        assert isinstance(get_model('decision_tree'), DecisionTreeModel)
        assert isinstance(get_model('random_forest'), RandomForestModel)

    def test_pass_through_instance(self):
        m = RandomForestModel()
        assert get_model(m) is m

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown model'):
            get_model('nonexistent')

    def test_bad_type(self):
        with pytest.raises(TypeError):
            get_model(123)

    def test_forwards_random_state(self):
        m = get_model('random_forest', random_state=42, n_estimators=5)
        assert m.random_state == 42
        assert m.n_estimators == 5


class TestRegisterModel:
    def test_register_custom(self):
        class Custom(UncertaintyModel):
            name = 'custom_model'

            def __init__(self, random_state=None):
                self.random_state = random_state
                self._mean = None

            def fit(self, X, y):
                self._mean = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(X.shape[0], self._mean)

            def predict_with_uncertainty(self, X):
                return self.predict(X), np.ones(X.shape[0])

        register_model('custom_test_model', Custom)
        m = get_model('custom_test_model', random_state=0)
        assert isinstance(m, Custom)
