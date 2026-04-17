"""Uncertainty-aware multiple imputation for tabular data.

A Python library implementing an iterative imputation framework guided by
uncertainty (acquisition) functions, as described in the paper:

    "Uncertainty-aware approach for multiple imputation using conventional
    and machine learning models: a real-world data study."

Top-level API
-------------

>>> from uncertainty_imputation import UncertaintyImputer, MCARAmputer
>>> imputer = UncertaintyImputer(model='xgboost', acquisition='ei',
...                              max_iter=10, random_state=42)
>>> imputer.fit(X_train)
>>> X_imputed = imputer.transform(X_missing)
"""

from .amputation import BaseAmputer, MARAmputer, MCARAmputer, MNARAmputer
from .imputation import (
    AcquisitionFunction,
    BaseImputer,
    DecisionTreeModel,
    ExpectedImprovement,
    LinearRegressionModel,
    MeanImputer,
    MICEImputer,
    ProbabilityOfImprovement,
    RandomForestModel,
    UncertaintyImputer,
    UncertaintyModel,
    UncertaintySampling,
    XGBoostModel,
    get_acquisition,
    get_model,
    register_acquisition,
    register_model,
)
from .utils import (
    evaluate_imputation,
    get_missing_mask,
    mae,
    missing_summary,
    normalized_rmse,
    per_column_rmse,
    rmse,
    standardize,
    unstandardize,
)

__version__ = '0.2.0'

__all__ = [
    '__version__',
    # Imputers
    'UncertaintyImputer',
    'MeanImputer',
    'MICEImputer',
    'BaseImputer',
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
    # Amputation
    'BaseAmputer',
    'MCARAmputer',
    'MARAmputer',
    'MNARAmputer',
    # Utilities
    'rmse',
    'mae',
    'normalized_rmse',
    'per_column_rmse',
    'evaluate_imputation',
    'get_missing_mask',
    'missing_summary',
    'standardize',
    'unstandardize',
]
