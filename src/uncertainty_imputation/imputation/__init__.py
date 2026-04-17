"""Uncertainty-aware iterative imputation."""

from .acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UncertaintySampling,
    get_acquisition,
    register_acquisition,
)
from .base import BaseImputer
from .baselines import MeanImputer, MICEImputer
from .iterative import UncertaintyImputer
from .models import (
    DecisionTreeModel,
    LinearRegressionModel,
    RandomForestModel,
    UncertaintyModel,
    XGBoostModel,
    get_model,
    register_model,
)

__all__ = [
    # Main class
    'UncertaintyImputer',
    # Base
    'BaseImputer',
    # Baselines
    'MeanImputer',
    'MICEImputer',
    # Models
    'UncertaintyModel',
    'LinearRegressionModel',
    'DecisionTreeModel',
    'RandomForestModel',
    'XGBoostModel',
    'get_model',
    'register_model',
    # Acquisition
    'AcquisitionFunction',
    'UncertaintySampling',
    'ProbabilityOfImprovement',
    'ExpectedImprovement',
    'get_acquisition',
    'register_acquisition',
]
