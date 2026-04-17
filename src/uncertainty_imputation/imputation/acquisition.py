"""Acquisition functions for uncertainty-aware imputation.

Each acquisition function scores candidate missing-value predictions by how
informative or promising they are, given a predictive mean and a predictive
standard deviation. Higher scores indicate higher priority / higher
information content.

All acquisition functions share the interface::

    score = acquisition(mean, std, y_best=None)

where ``mean`` and ``std`` are 1D arrays of the same length, and ``y_best`` is
the current best (lowest-loss) target value used by Probability of Improvement
and Expected Improvement. The functions are vectorised and return a 1D numpy
array of scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.stats import norm


class AcquisitionFunction(ABC):
    '''Base class for acquisition functions.

    Subclasses implement :meth:`score`, which returns a per-sample score given
    predictive means and standard deviations.
    '''

    name: str = 'base'

    @abstractmethod
    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: Optional[float] = None,
    ) -> np.ndarray:
        '''Compute the acquisition score for each candidate prediction.

        Parameters
        ----------
        mean : np.ndarray of shape (n_samples,)
            Predictive mean for each candidate.
        std : np.ndarray of shape (n_samples,)
            Predictive standard deviation for each candidate.
        y_best : float, optional
            Current best target value (used by PI and EI).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Acquisition score for each candidate.
        '''

    def __call__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: Optional[float] = None,
    ) -> np.ndarray:
        mean = np.asarray(mean, dtype=float).ravel()
        std = np.asarray(std, dtype=float).ravel()
        if mean.shape != std.shape:
            raise ValueError(
                f'mean and std must have the same shape, got '
                f'{mean.shape} and {std.shape}'
            )
        # Guard against exactly-zero std which would produce NaNs in PI/EI
        std = np.clip(std, a_min=1e-12, a_max=None)
        return self.score(mean, std, y_best=y_best)


class UncertaintySampling(AcquisitionFunction):
    '''Uncertainty Sampling: score equals predictive standard deviation.

    The highest-uncertainty candidates are prioritised, which is the classical
    active-learning / Bayesian-optimisation exploration criterion.
    '''

    name = 'uncertainty_sampling'

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: Optional[float] = None,
    ) -> np.ndarray:
        return std


class ProbabilityOfImprovement(AcquisitionFunction):
    '''Probability of Improvement over the current best observation.

    For a minimisation target (lower is better), PI is defined as

    .. math:: \\mathrm{PI}(x) = \\Phi\\!\\left(\\frac{y^* - \\mu(x) - \\xi}{\\sigma(x)}\\right)

    where :math:`\\Phi` is the standard normal CDF, :math:`y^*` is the current
    best observed target, :math:`\\mu` and :math:`\\sigma` are the predictive
    mean and std, and :math:`\\xi \\ge 0` is an exploration parameter.
    '''

    name = 'probability_of_improvement'

    def __init__(self, xi: float = 0.01) -> None:
        self.xi = float(xi)

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: Optional[float] = None,
    ) -> np.ndarray:
        if y_best is None:
            # Without a best target, fall back to uncertainty sampling so the
            # function stays well-defined during the first imputation pass.
            return std
        z = (y_best - mean - self.xi) / std
        return norm.cdf(z)


class ExpectedImprovement(AcquisitionFunction):
    '''Expected Improvement over the current best observation.

    For a minimisation target,

    .. math:: \\mathrm{EI}(x) = (y^* - \\mu(x) - \\xi)\\,\\Phi(z) + \\sigma(x)\\,\\phi(z)

    with :math:`z = (y^* - \\mu(x) - \\xi) / \\sigma(x)`. :math:`\\Phi` and
    :math:`\\phi` are the standard-normal CDF and PDF.
    '''

    name = 'expected_improvement'

    def __init__(self, xi: float = 0.01) -> None:
        self.xi = float(xi)

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: Optional[float] = None,
    ) -> np.ndarray:
        if y_best is None:
            return std
        improvement = y_best - mean - self.xi
        z = improvement / std
        return improvement * norm.cdf(z) + std * norm.pdf(z)


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_ACQUISITION_REGISTRY: dict[str, type[AcquisitionFunction]] = {
    'us': UncertaintySampling,
    'uncertainty_sampling': UncertaintySampling,
    'pi': ProbabilityOfImprovement,
    'probability_of_improvement': ProbabilityOfImprovement,
    'ei': ExpectedImprovement,
    'expected_improvement': ExpectedImprovement,
}


def get_acquisition(
    acquisition: 'str | AcquisitionFunction',
    **kwargs,
) -> AcquisitionFunction:
    '''Resolve an acquisition function from a string alias or instance.

    Parameters
    ----------
    acquisition : str or AcquisitionFunction
        Either an alias (``'us'``, ``'pi'``, ``'ei'`` and their long forms) or
        an already-constructed :class:`AcquisitionFunction` instance.
    **kwargs
        Extra keyword arguments passed to the acquisition constructor when a
        string alias is provided (e.g. ``xi`` for PI and EI).

    Returns
    -------
    AcquisitionFunction
    '''
    if isinstance(acquisition, AcquisitionFunction):
        return acquisition
    if not isinstance(acquisition, str):
        raise TypeError(
            'acquisition must be a string alias or AcquisitionFunction '
            f'instance, got {type(acquisition).__name__}'
        )
    key = acquisition.lower()
    if key not in _ACQUISITION_REGISTRY:
        available = sorted(set(_ACQUISITION_REGISTRY))
        raise ValueError(
            f'Unknown acquisition {acquisition!r}. Available: {available}'
        )
    return _ACQUISITION_REGISTRY[key](**kwargs)


def register_acquisition(
    name: str,
    cls: type[AcquisitionFunction],
) -> None:
    '''Register a custom acquisition function under a string alias.'''
    if not issubclass(cls, AcquisitionFunction):
        raise TypeError('cls must be a subclass of AcquisitionFunction')
    _ACQUISITION_REGISTRY[name.lower()] = cls


__all__ = [
    'AcquisitionFunction',
    'UncertaintySampling',
    'ProbabilityOfImprovement',
    'ExpectedImprovement',
    'get_acquisition',
    'register_acquisition',
]
