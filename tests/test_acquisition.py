"""Tests for acquisition functions."""

import numpy as np
import pytest

from uncertainty_imputation.imputation.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UncertaintySampling,
    get_acquisition,
    register_acquisition,
)


class TestUncertaintySampling:
    def test_score_equals_std(self):
        acq = UncertaintySampling()
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.5, 0.2])
        result = acq(mean, std)
        # std gets clipped at 1e-12 then returned
        np.testing.assert_allclose(result, std, atol=1e-10)

    def test_ignores_y_best(self):
        acq = UncertaintySampling()
        mean = np.array([1.0, 2.0])
        std = np.array([0.3, 0.4])
        result_none = acq(mean, std, y_best=None)
        result_with = acq(mean, std, y_best=0.5)
        np.testing.assert_array_equal(result_none, result_with)


class TestProbabilityOfImprovement:
    def test_returns_std_when_no_y_best(self):
        # Falls back to uncertainty sampling
        acq = ProbabilityOfImprovement()
        std = np.array([0.1, 0.5])
        result = acq(np.array([1.0, 2.0]), std, y_best=None)
        np.testing.assert_allclose(result, std, atol=1e-10)

    def test_pi_in_unit_interval(self):
        acq = ProbabilityOfImprovement(xi=0.0)
        mean = np.array([0.0, 5.0, 10.0])
        std = np.array([1.0, 1.0, 1.0])
        result = acq(mean, std, y_best=5.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_pi_higher_for_better_mean(self):
        # For minimisation: smaller mean -> higher PI
        acq = ProbabilityOfImprovement(xi=0.0)
        std = np.array([1.0, 1.0])
        mean = np.array([0.0, 10.0])
        result = acq(mean, std, y_best=5.0)
        assert result[0] > result[1]


class TestExpectedImprovement:
    def test_returns_std_when_no_y_best(self):
        acq = ExpectedImprovement()
        std = np.array([0.3, 0.7])
        result = acq(np.array([1.0, 2.0]), std, y_best=None)
        np.testing.assert_allclose(result, std, atol=1e-10)

    def test_ei_non_negative(self):
        # EI should always be non-negative
        acq = ExpectedImprovement(xi=0.0)
        rng = np.random.default_rng(0)
        mean = rng.normal(size=50)
        std = np.abs(rng.normal(size=50)) + 0.01
        result = acq(mean, std, y_best=0.0)
        assert np.all(result >= -1e-10)

    def test_ei_higher_when_improvement_likely(self):
        acq = ExpectedImprovement(xi=0.0)
        std = np.array([1.0, 1.0])
        # mean=0 well below y_best=5 (strong improvement) vs mean=10 above it
        result = acq(np.array([0.0, 10.0]), std, y_best=5.0)
        assert result[0] > result[1]


class TestGetAcquisition:
    def test_string_aliases(self):
        assert isinstance(get_acquisition('us'), UncertaintySampling)
        assert isinstance(get_acquisition('uncertainty_sampling'), UncertaintySampling)
        assert isinstance(get_acquisition('pi'), ProbabilityOfImprovement)
        assert isinstance(get_acquisition('ei'), ExpectedImprovement)
        assert isinstance(get_acquisition('EI'), ExpectedImprovement)  # case-insensitive

    def test_pass_through_instance(self):
        acq = UncertaintySampling()
        assert get_acquisition(acq) is acq

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown acquisition'):
            get_acquisition('nonexistent')

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            get_acquisition(42)

    def test_forwards_kwargs(self):
        acq = get_acquisition('ei', xi=0.5)
        assert isinstance(acq, ExpectedImprovement)
        assert acq.xi == 0.5


class TestRegisterAcquisition:
    def test_register_custom(self):
        class Custom(AcquisitionFunction):
            name = 'custom_acq'

            def score(self, mean, std, y_best=None):
                return std * 2

        register_acquisition('custom_test', Custom)
        acq = get_acquisition('custom_test')
        assert isinstance(acq, Custom)

    def test_reject_non_subclass(self):
        class NotAnAcq:
            pass

        with pytest.raises(TypeError):
            register_acquisition('bad', NotAnAcq)


class TestInputValidation:
    def test_shape_mismatch(self):
        acq = UncertaintySampling()
        with pytest.raises(ValueError, match='same shape'):
            acq(np.array([1.0, 2.0]), np.array([0.1]))

    def test_zero_std_clipped(self):
        # std=0 would produce div-by-zero in PI/EI; we clip it
        acq = ProbabilityOfImprovement()
        result = acq(np.array([1.0]), np.array([0.0]), y_best=5.0)
        assert np.isfinite(result).all()
