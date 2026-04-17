"""Iterative uncertainty-aware imputation.

The :class:`UncertaintyImputer` iteratively refines missing values in a
tabular dataset. At each iteration and for each column with missing values:

1. Rows where the column is observed are used as training data; rows where
   it is missing are used as prediction targets.
2. The chosen uncertainty-aware model predicts the missing values along with
   a per-prediction standard deviation.
3. An acquisition function converts ``(mean, std)`` into a per-prediction
   score which is recorded for downstream analysis.
4. Missing entries are filled with the predicted means.

The process repeats for ``max_iter`` iterations, or until changes between
iterations fall below a tolerance. A simple mean initialisation is used for
the first pass so that models always see complete feature matrices.

The scheme is a generalisation of the MICE paradigm: instead of drawing
samples from a posterior, we exploit the acquisition score as an auxiliary
quantity that can be used to (a) weight updates, (b) prioritise columns, or
(c) expose calibration / reliability information to downstream users.
'''
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .acquisition import AcquisitionFunction, get_acquisition
from .base import ArrayLike, BaseImputer
from .models import UncertaintyModel, get_model


class UncertaintyImputer(BaseImputer):
    '''Iterative uncertainty-aware imputer.

    Parameters
    ----------
    model : str or UncertaintyModel, default ``'random_forest'``
        Per-feature regressor. String aliases: ``'linear'``,
        ``'decision_tree'``, ``'random_forest'``, ``'xgboost'``.
    acquisition : str or AcquisitionFunction, default ``'us'``
        Acquisition function. String aliases: ``'us'``, ``'pi'``, ``'ei'``.
    max_iter : int, default 10
        Maximum number of imputation passes over the feature set.
    tol : float, default 1e-3
        Convergence tolerance on the maximum relative change in imputed
        values between two consecutive iterations. If exceeded, iteration
        stops early.
    initial_strategy : {'mean', 'median'}, default ``'mean'``
        Initialisation strategy for missing values before the first pass.
    column_order : {'ascending', 'descending', 'random'}, default ``'ascending'``
        Order in which columns with missing values are visited within an
        iteration. ``'ascending'`` visits the column with the *fewest*
        missing values first (standard MICE choice).
    min_obs_for_model : int, default 10
        If a column has fewer than this many observed values, it is filled
        with the initialisation strategy (a model is not trained for it).
    update_rule : {'mean', 'acquisition_weighted', 'sample'}, default ``'acquisition_weighted'``
        How the imputed value is derived from the model's ``(mean, std)``
        prediction and the acquisition score.

        * ``'mean'`` — fill with the predictive mean, regardless of
          acquisition. The acquisition score is recorded in
          ``uncertainty_`` but does not alter the update. All three
          acquisition functions therefore produce identical imputations;
          only the recorded scores differ.
        * ``'acquisition_weighted'`` (default) — interpolate between the
          previous iterate and the predictive mean by a weight ``α`` in
          ``[0, 1]`` derived from the acquisition score. High-acquisition
          cells move aggressively toward the model prediction; low-
          acquisition cells remain close to their previous value. Weights
          are min-max normalised within each column's missing cells, so
          each acquisition function produces a genuinely different update
          trajectory.
        * ``'sample'`` — draw the imputed value from
          ``N(mean, std)``. Recovers the classical "proper multiple
          imputation" semantics of MICE.
    predictor_columns : dict, optional
        Restrict which columns are used as predictors when imputing each
        target column. Keys are target column identifiers (string names for
        DataFrames, integer indices for numpy arrays, or either for
        DataFrames); values are sequences of predictor column identifiers.

        Example::

            # "impute income using only age and education"
            # "impute bmi using only age, height and weight"
            # any other column with missingness falls back to 'all others'
            UncertaintyImputer(
                predictor_columns={
                    'income': ['age', 'education'],
                    'bmi':    ['age', 'height', 'weight'],
                },
            )

        If ``None`` (default), every column except the target itself is
        used as a predictor (the fully-conditional MICE specification).
        A target that is its own listed predictor is silently removed from
        its predictor set — a column cannot predict itself. If a target
        ends up with an empty predictor set, its missing cells are filled
        with the initialisation value for that column.
    random_state : int, optional
        Seed controlling model initialisation and any tie-breaking.
    verbose : bool, default False
        If True, print per-iteration progress information.
    model_kwargs : dict, optional
        Extra keyword arguments forwarded to the model constructor.
    acquisition_kwargs : dict, optional
        Extra keyword arguments forwarded to the acquisition constructor.

    Attributes
    ----------
    n_iter_ : int
        Number of iterations actually performed.
    history_ : list of dict
        Per-iteration diagnostics including mean/std/change per column.
    uncertainty_ : dict[str or int, np.ndarray]
        Final per-missing-cell acquisition scores, keyed by column.
    '''

    def __init__(
        self,
        model: Union[str, UncertaintyModel] = 'random_forest',
        acquisition: Union[str, AcquisitionFunction] = 'us',
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = 'mean',
        column_order: str = 'ascending',
        min_obs_for_model: int = 10,
        update_rule: str = 'acquisition_weighted',
        predictor_columns: Optional[dict] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        model_kwargs: Optional[dict] = None,
        acquisition_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.acquisition = acquisition
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.initial_strategy = initial_strategy
        self.column_order = column_order
        self.min_obs_for_model = int(min_obs_for_model)
        self.update_rule = update_rule
        self.predictor_columns = (
            dict(predictor_columns) if predictor_columns else None
        )
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.model_kwargs = dict(model_kwargs) if model_kwargs else {}
        self.acquisition_kwargs = dict(acquisition_kwargs) if acquisition_kwargs else {}

        # Fitted state
        self.n_iter_: int = 0
        self.history_: list[dict] = []
        self.uncertainty_: dict = {}
        self._initial_values_: Optional[np.ndarray] = None  # shape (n_features,)
        self._fitted_columns_: Optional[list[str]] = None
        self._predictor_map_: Optional[dict[int, list[int]]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _initial_fill(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        '''Return initial fill values (per column) using the chosen strategy.'''
        n_features = X.shape[1]
        fills = np.zeros(n_features)
        for j in range(n_features):
            observed = X[~mask[:, j], j]
            if observed.size == 0:
                fills[j] = 0.0
                continue
            if self.initial_strategy == 'mean':
                fills[j] = float(np.mean(observed))
            elif self.initial_strategy == 'median':
                fills[j] = float(np.median(observed))
            else:
                raise ValueError(
                    f"initial_strategy must be 'mean' or 'median', "
                    f'got {self.initial_strategy!r}'
                )
        return fills

    def _column_visit_order(
        self, mask: np.ndarray, rng: np.random.Generator
    ) -> list[int]:
        '''Return the order in which columns should be imputed.'''
        missing_counts = mask.sum(axis=0)
        cols_with_missing = np.where(missing_counts > 0)[0]
        if self.column_order == 'ascending':
            order = cols_with_missing[np.argsort(missing_counts[cols_with_missing])]
        elif self.column_order == 'descending':
            order = cols_with_missing[np.argsort(-missing_counts[cols_with_missing])]
        elif self.column_order == 'random':
            order = rng.permutation(cols_with_missing)
        else:
            raise ValueError(
                "column_order must be 'ascending', 'descending', or 'random', "
                f'got {self.column_order!r}'
            )
        return list(map(int, order))

    def _resolve_predictor_map(
        self, columns: Optional[list[str]]
    ) -> Optional[dict[int, list[int]]]:
        '''Convert ``self.predictor_columns`` (name- or index-keyed) to a
        dict of integer-index → list[int].

        Called once at fit time and stored in ``self._predictor_map_``.
        Returns ``None`` if the user did not specify any predictor
        restrictions, in which case ``_impute_once`` uses the
        "all-other-columns" default.
        '''
        if self.predictor_columns is None:
            return None

        n_features = self._n_features
        name_to_idx: dict = {}
        if columns is not None:
            name_to_idx = {name: i for i, name in enumerate(columns)}

        def _resolve_one(key, context: str) -> int:
            if isinstance(key, (int, np.integer)):
                idx = int(key)
                if not 0 <= idx < n_features:
                    raise ValueError(
                        f'{context}: column index {idx} out of range for '
                        f'a {n_features}-feature matrix'
                    )
                return idx
            if isinstance(key, str):
                if columns is None:
                    raise TypeError(
                        f'{context}: string column name {key!r} requires a '
                        'pandas DataFrame input; got numpy array'
                    )
                if key not in name_to_idx:
                    raise KeyError(
                        f'{context}: column name {key!r} not in DataFrame '
                        f'columns {list(name_to_idx)}'
                    )
                return name_to_idx[key]
            raise TypeError(
                f'{context}: column identifier must be int or str, got '
                f'{type(key).__name__}'
            )

        out: dict[int, list[int]] = {}
        for target, preds in self.predictor_columns.items():
            target_idx = _resolve_one(target, 'predictor_columns key')
            pred_idxs: list[int] = []
            seen: set[int] = set()
            for p in preds:
                p_idx = _resolve_one(p, f'predictor_columns[{target!r}]')
                if p_idx == target_idx:
                    # A column can't predict itself. Silently drop.
                    continue
                if p_idx in seen:
                    continue
                seen.add(p_idx)
                pred_idxs.append(p_idx)
            out[target_idx] = pred_idxs
        return out

    def _predictors_for(self, j: int, n_features: int) -> list[int]:
        '''Return the ordered list of predictor column indices for target ``j``.'''
        if self._predictor_map_ is not None and j in self._predictor_map_:
            return self._predictor_map_[j]
        return [k for k in range(n_features) if k != j]

    def _impute_once(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict]:
        '''Run one full pass over all columns with missing values.

        Returns the updated X matrix and a per-column diagnostics dict.
        '''
        X_new = X.copy()
        per_column_diag: dict[int, dict] = {}
        uncertainty: dict[int, np.ndarray] = {}

        for j in self._column_visit_order(mask, rng):
            col_mask = mask[:, j]
            n_missing = int(col_mask.sum())
            n_observed = int((~col_mask).sum())
            if n_missing == 0:
                continue
            if n_observed < self.min_obs_for_model:
                # Not enough data; keep initialisation.
                fill_val = float(np.nanmean(X_new[~col_mask, j])) \
                    if n_observed > 0 else 0.0
                X_new[col_mask, j] = fill_val
                continue

            # Features = the user-specified predictor columns (or all other
            # columns if none specified). Predictors are already filled from
            # prior passes / initialisation, so X_train has no NaNs.
            other_cols = self._predictors_for(j, X_new.shape[1])
            if not other_cols:
                # No predictors available (e.g. user specified an empty list
                # or a single-column matrix); keep the initialisation fill.
                fill_val = float(self._initial_values_[j]) \
                    if self._initial_values_ is not None else 0.0
                X_new[col_mask, j] = fill_val
                continue
            X_train = X_new[~col_mask][:, other_cols]
            y_train = X_new[~col_mask, j]
            X_pred = X_new[col_mask][:, other_cols]

            # Build a fresh model each pass. Training is cheap relative to
            # the cost of incorrect early fits contaminating later columns.
            model = get_model(
                self.model,
                random_state=self.random_state,
                **self.model_kwargs,
            )
            model.fit(X_train, y_train)
            mean, std = model.predict_with_uncertainty(X_pred)

            # Acquisition score: y_best is the min of observed training targets
            # (standard minimisation convention).
            acquisition = get_acquisition(
                self.acquisition, **self.acquisition_kwargs
            )
            y_best = float(np.min(y_train)) if y_train.size else None
            scores = acquisition(mean, std, y_best=y_best)

            # Derive the imputed value from mean, std, and acquisition score
            # according to the configured update rule.
            previous = X_new[col_mask, j]
            imputed = self._apply_update_rule(
                previous=previous, mean=mean, std=std, scores=scores, rng=rng,
            )
            X_new[col_mask, j] = imputed
            uncertainty[j] = scores
            per_column_diag[j] = dict(
                n_missing=n_missing,
                mean=float(np.mean(mean)),
                std=float(np.mean(std)),
                acq_mean=float(np.mean(scores)),
            )

        return X_new, {'columns': per_column_diag, 'uncertainty': uncertainty}

    def _apply_update_rule(
        self,
        previous: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        '''Combine ``(previous, mean, std, scores)`` into the imputed value.

        See the ``update_rule`` parameter of :class:`UncertaintyImputer` for
        the semantics of each rule.
        '''
        if self.update_rule == 'mean':
            return mean

        if self.update_rule == 'sample':
            return mean + std * rng.standard_normal(size=mean.shape)

        if self.update_rule == 'acquisition_weighted':
            # Min-max normalise the scores within this column's missing cells
            # to get per-cell weights in [0, 1]. Constant scores collapse to
            # weight 1 (fall back to pure-mean update), which mirrors the
            # 'mean' rule when the acquisition provides no differentiation.
            if scores.size == 0:
                return mean
            s_min = float(np.min(scores))
            s_max = float(np.max(scores))
            if s_max - s_min < 1e-12:
                return mean
            alpha = (scores - s_min) / (s_max - s_min)
            return previous + alpha * (mean - previous)

        raise ValueError(
            "update_rule must be 'mean', 'acquisition_weighted', or "
            f"'sample', got {self.update_rule!r}"
        )

    def _relative_change(
        self,
        X_old: np.ndarray,
        X_new: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        '''Mean absolute change across imputed cells, normalised by column scale.

        For each column with missing values, we compute ``mean(|X_new - X_old|)``
        over that column's imputed cells and divide by the column's standard
        deviation (computed on all rows, post-imputation, with a small floor).
        The returned value is the maximum of this per-column ratio. This
        metric is scale-invariant, robust to columns with values near zero,
        and matches the convention used by ``sklearn.impute.IterativeImputer``.
        '''
        if not mask.any():
            return 0.0
        max_ratio = 0.0
        for j in range(X_new.shape[1]):
            col_mask = mask[:, j]
            if not col_mask.any():
                continue
            mean_abs_diff = float(np.mean(np.abs(X_new[col_mask, j] - X_old[col_mask, j])))
            col_std = float(np.std(X_new[:, j]))
            denom = max(col_std, 1e-8)
            max_ratio = max(max_ratio, mean_abs_diff / denom)
        return max_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, y=None) -> 'UncertaintyImputer':
        '''Fit the imputer by running the iterative procedure on ``X``.

        The final imputed matrix is computed during ``fit`` and stored so
        ``transform(X)`` on the same data returns it without re-running
        (similar to scikit-learn's ``IterativeImputer``).
        '''
        arr, columns, index = self._to_array(X)
        mask = np.isnan(arr)

        self._fitted_columns_ = columns
        self._n_features = arr.shape[1]
        self._predictor_map_ = self._resolve_predictor_map(columns)
        self._initial_values_ = self._initial_fill(arr, mask)

        # Apply initial fill
        X_current = arr.copy()
        for j in range(X_current.shape[1]):
            X_current[mask[:, j], j] = self._initial_values_[j]

        rng = np.random.default_rng(self.random_state)
        self.history_ = []
        self.uncertainty_ = {}
        self.n_iter_ = 0

        for it in range(1, self.max_iter + 1):
            X_prev = X_current.copy()
            X_current, diag = self._impute_once(X_current, mask, rng)
            change = self._relative_change(X_prev, X_current, mask)
            self.history_.append(
                dict(iteration=it, change=change, **diag)
            )
            self.uncertainty_ = diag['uncertainty']
            self.n_iter_ = it
            if self.verbose:
                print(
                    f'[UncertaintyImputer] iter={it} '
                    f'max_rel_change={change:.6f}'
                )
            if change < self.tol:
                break

        self._fit_X_ = X_current  # cached for fit_transform on same data
        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        '''Impute missing values in ``X`` using the fitted imputer.

        If ``X`` is the same matrix passed to :meth:`fit` (by shape and mask
        pattern), the cached result is returned. Otherwise, the iterative
        procedure is re-run on ``X`` using the saved initialisation values.
        '''
        self._check_is_fitted()
        arr, columns, index = self._to_array(X)
        mask = np.isnan(arr)

        if arr.shape[1] != self._n_features:
            raise ValueError(
                f'X has {arr.shape[1]} features but imputer was fit with '
                f'{self._n_features} features'
            )

        # Fast path: if this looks like the same matrix used during fit
        # (same shape, same mask, same observed values), return the cached
        # imputed matrix instead of re-running the iteration.
        if (
            arr.shape == self._fit_X_.shape
            and mask.sum() == int(np.isnan(arr).sum())
            and np.array_equal(mask, np.isnan(arr))  # same mask pattern
            and np.allclose(arr[~mask], self._fit_X_[~mask])
        ):
            return self._wrap_output(self._fit_X_.copy(), columns, index)

        # Otherwise re-run iteration on new data, re-using initial values for
        # columns seen at fit time.
        X_current = arr.copy()
        for j in range(X_current.shape[1]):
            if mask[:, j].any():
                X_current[mask[:, j], j] = self._initial_values_[j]

        rng = np.random.default_rng(self.random_state)
        for _ in range(self.max_iter):
            X_prev = X_current.copy()
            X_current, _ = self._impute_once(X_current, mask, rng)
            if self._relative_change(X_prev, X_current, mask) < self.tol:
                break

        return self._wrap_output(X_current, columns, index)

    def fit_transform(self, X: ArrayLike, y=None) -> ArrayLike:
        self.fit(X, y=y)
        # Return cached result in original wrapper.
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(self._fit_X_.copy(), columns=X.columns, index=X.index)
        return self._fit_X_.copy()


def _hash_array_shape(arr: np.ndarray, mask: np.ndarray) -> tuple:
    '''Deprecated helper retained for backwards compatibility.'''
    return (arr.shape, int(mask.sum()), float(np.nansum(arr)))


__all__ = ['UncertaintyImputer']
