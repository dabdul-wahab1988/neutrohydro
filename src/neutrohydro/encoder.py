"""
NDG Encoder (Neutrosophic Data Generator) for NeutroHydro.

Implements Section 3 of the specification:
- Maps scalar values to neutrosophic triplets (T, I, F)
- T (Truth): Baseline component via robust baseline operators
- I (Indeterminacy): Uncertainty/ambiguity channel
- F (Falsity): Perturbation likelihood from residuals

Out-of-sample behaviour
-----------------------
``fit`` learns the baseline *operator* (column medians, group medians, or a
low-rank subspace) and ``transform`` applies that fitted operator to any
data. Earlier versions re-estimated the baseline on whatever batch was
passed to ``transform``, so a prediction for new samples depended on the
other samples in the batch (and a uniformly shifted batch looked
unperturbed).

Indeterminacy
-------------
The default I channel is the local heterogeneity of each ion among the
``n_neighbors`` most similar *training* samples (nearest neighbours in the
standardized ion space). The older rolling window over row order is still
available as ``indeterminacy="rolling"``. Use it only when rows have a
meaningful order (time series, a transect); for unordered samples it makes
the results depend on the order of rows in the input file.

For an encoding whose channels are provably non-overlapping (exact
attribution), see :mod:`neutrohydro.orthogonal`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.spatial import cKDTree


class BaselineType(Enum):
    """Types of baseline operators for the Truth channel."""

    MEDIAN = "median"                    # Robust columnwise median
    HYDROFACIES_MEDIAN = "hydrofacies"   # Median conditioned by groups
    LOW_RANK = "low_rank"                # Truncated SVD baseline
    ROBUST_PCA = "robust_pca"            # Low-rank + sparse decomposition


class FalsityMap(Enum):
    """Monotone maps for converting residuals to falsity [0,1]."""

    EXPONENTIAL = "exponential"  # 1 - exp(-u)
    LOGISTIC = "logistic"        # 1 / (1 + exp(-a*(u-b)))


class IndeterminacyMode(Enum):
    """Default constructions of the Indeterminacy channel."""

    KNN = "knn"          # Local heterogeneity among nearest training samples
    ROLLING = "rolling"  # Legacy: rolling window over row order
    NONE = "none"        # I = 0


@dataclass
class EncoderParams:
    """Stores encoder parameters and metadata."""

    baseline_type: BaselineType
    baseline_rank: Optional[int]  # For low-rank methods
    falsity_map: FalsityMap
    falsity_params: dict          # Parameters for falsity map (a, b for logistic)
    sigma: NDArray[np.floating]   # Robust per-variable residual scales
    baseline_metadata: dict       # Additional baseline info (e.g., group means)


@dataclass
class TripletData:
    """
    Container for neutrosophic triplet channels.

    For the legacy NDG encoding (``encoding="ndg"``) T is the baseline and
    I, F are memberships in [0, 1]. For the Jacobian-orthogonal encoding
    (``encoding="orthogonal"``) T, I and F are signed, mutually orthogonal
    components with ``T + I + F = X_std - center``; memberships are then
    stored in ``memberships``.
    """

    T: NDArray[np.floating]  # Truth channel (baseline)
    I: NDArray[np.floating]  # Indeterminacy channel (ambiguity)
    F: NDArray[np.floating]  # Falsity channel (perturbation)
    R: NDArray[np.floating]  # Residuals (X_std - T)
    encoding: str = "ndg"    # "ndg" (legacy) or "orthogonal"
    sample_weights: Optional[NDArray[np.floating]] = None  # Row weights
    memberships: Optional[dict] = None  # Orthogonal encoding memberships


def _fill_nan_columns(
    X: NDArray[np.floating],
    fill: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Replace NaNs columnwise with ``fill`` values."""
    if not np.any(np.isnan(X)):
        return X
    X = X.copy()
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = fill[cols]
    return X


class NDGEncoder:
    """
    Neutrosophic Data Generator Encoder.

    Maps standardized predictor matrix X_std to triplet channels (T, I, F)
    following the mathematical framework in Section 3.

    Parameters
    ----------
    baseline_type : BaselineType or str, default=BaselineType.MEDIAN
        Type of baseline operator for Truth channel:
        - "median": Robust columnwise median (default)
        - "hydrofacies": Median conditioned by groups
        - "low_rank": Truncated SVD
        - "robust_pca": Low-rank + sparse decomposition

    baseline_rank : int, optional
        Rank for low-rank baseline methods. Required if baseline_type
        is LOW_RANK or ROBUST_PCA.

    falsity_map : FalsityMap or str, default=FalsityMap.EXPONENTIAL
        Monotone map for Falsity channel:
        - "exponential": F = 1 - exp(-u)
        - "logistic": F = 1 / (1 + exp(-a*(u-b)))

    falsity_params : dict, optional
        Parameters for falsity map. For logistic: {"a": float, "b": float}.
        Default for exponential: {}
        Default for logistic: {"a": 2.0, "b": 1.0}

    delta : float, default=1e-10
        Small constant for numerical stability.

    indeterminacy : IndeterminacyMode or str, default="knn"
        Default construction of the I channel when no
        ``indeterminacy_func`` is passed to ``transform``:
        - "knn": local variance of each ion among the ``n_neighbors``
          nearest training samples (order-invariant)
        - "rolling": legacy rolling window over row order
        - "none": I = 0

    n_neighbors : int, default=5
        Neighbourhood size for the "knn" and "rolling" indeterminacy.

    Attributes
    ----------
    params_ : EncoderParams
        Fitted encoder parameters.
    is_fitted_ : bool
        Whether the encoder has been fitted.

    Notes
    -----
    The median baseline is constant across samples, so its T channel
    carries no sample-to-sample information and cannot explain variation
    in the target (pi_G is then 0 for every ion). Use a low-rank baseline,
    or the orthogonal encoder, when baseline attribution matters.
    """

    def __init__(
        self,
        baseline_type: BaselineType | str = BaselineType.MEDIAN,
        baseline_rank: Optional[int] = None,
        falsity_map: FalsityMap | str = FalsityMap.EXPONENTIAL,
        falsity_params: Optional[dict] = None,
        delta: float = 1e-10,
        indeterminacy: IndeterminacyMode | str = IndeterminacyMode.KNN,
        n_neighbors: int = 5,
    ):
        if isinstance(baseline_type, str):
            baseline_type = BaselineType(baseline_type)
        if isinstance(falsity_map, str):
            falsity_map = FalsityMap(falsity_map)
        if isinstance(indeterminacy, str):
            indeterminacy = IndeterminacyMode(indeterminacy)

        self.baseline_type = baseline_type
        self.baseline_rank = baseline_rank
        self.falsity_map = falsity_map
        self.delta = delta
        self.indeterminacy = indeterminacy
        self.n_neighbors = n_neighbors

        # Set default falsity parameters
        if falsity_params is None:
            if falsity_map == FalsityMap.LOGISTIC:
                falsity_params = {"a": 2.0, "b": 1.0}
            else:
                falsity_params = {}
        self.falsity_params = falsity_params

        self.params_: Optional[EncoderParams] = None
        self.is_fitted_: bool = False

        # Fitted baseline state (see `fit`)
        self._state: dict = {}

        # Validate rank requirement
        if baseline_type in [BaselineType.LOW_RANK, BaselineType.ROBUST_PCA]:
            if baseline_rank is None:
                raise ValueError(
                    f"baseline_rank required for {baseline_type.value}"
                )

    # ------------------------------------------------------------------
    # Baseline estimation (training data)
    # ------------------------------------------------------------------

    def _compute_baseline_median(
        self,
        X_std: NDArray[np.floating],
        groups: Optional[NDArray[np.integer]] = None,
    ) -> tuple[NDArray[np.floating], dict]:
        """Compute median baseline, optionally conditioned by groups."""
        n, p = X_std.shape

        if groups is None:
            # Global columnwise median replicated per row
            col_medians = np.nanmedian(X_std, axis=0)
            X_T = np.tile(col_medians, (n, 1))
            metadata = {"type": "global_median", "medians": col_medians}
        else:
            # Hydrofacies-conditioned median
            unique_groups = np.unique(groups)
            X_T = np.zeros_like(X_std)
            group_medians = {}

            for g in unique_groups:
                mask = groups == g
                group_data = X_std[mask, :]
                gm = np.nanmedian(group_data, axis=0)
                group_medians[int(g)] = gm
                X_T[mask, :] = gm

            metadata = {
                "type": "hydrofacies_median",
                "group_medians": group_medians,
                "global_medians": np.nanmedian(X_std, axis=0),
            }

        return X_T, metadata

    def _compute_baseline_low_rank(
        self,
        X_std: NDArray[np.floating],
        rank: int,
    ) -> tuple[NDArray[np.floating], dict]:
        """Compute low-rank baseline via truncated SVD."""
        # Handle missing values with mean imputation for SVD
        col_means = np.nanmean(X_std, axis=0)
        X_work = _fill_nan_columns(X_std, col_means)

        # Truncated SVD
        U, s, Vt = svd(X_work, full_matrices=False)

        # Reconstruct with top-r components
        r = min(rank, len(s))
        X_T = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]

        metadata = {
            "type": "low_rank",
            "rank": r,
            "singular_values": s[:r],
            "explained_variance_ratio": (s[:r] ** 2).sum() / (s ** 2).sum(),
            # Fitted operator: T(x) = x V_r V_r^T
            "basis": Vt[:r, :].T,
            "fill_values": col_means,
        }

        return X_T, metadata

    def _compute_baseline_robust_pca(
        self,
        X_std: NDArray[np.floating],
        rank: int,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> tuple[NDArray[np.floating], dict]:
        """
        Compute robust PCA baseline (L + S decomposition).

        Uses iterative thresholding: X = L + S where L is low-rank and S is sparse.
        (The cap was 100 iterations in earlier versions, which typical
        groundwater tables need 100-300 iterations to reach; an unconverged
        L is not a fixed point of the decomposition.)
        """
        col_means = np.nanmean(X_std, axis=0)
        X_work = _fill_nan_columns(X_std, col_means)

        n, p = X_work.shape
        lam = 1.0 / np.sqrt(max(n, p))  # Regularization parameter

        L = np.zeros_like(X_work)
        S = np.zeros_like(X_work)
        r = min(rank, n, p)
        Vt = np.zeros((r, p))

        converged = False
        for iteration in range(max_iter):
            L_old = L.copy()

            # Update L (low-rank via SVD thresholding)
            U, s, Vt_full = svd(X_work - S, full_matrices=False)
            r = min(rank, len(s))
            Vt = Vt_full[:r, :]
            L = U[:, :r] @ np.diag(s[:r]) @ Vt

            # Update S (sparse via soft thresholding)
            residual = X_work - L
            S = np.sign(residual) * np.maximum(np.abs(residual) - lam, 0)

            # Check convergence
            if np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-10) < tol:
                converged = True
                break

        if not converged:
            warnings.warn(
                f"Robust PCA baseline did not converge in {max_iter} iterations.",
                stacklevel=3,
            )

        metadata = {
            "type": "robust_pca",
            "rank": r,
            "iterations": iteration + 1,
            "converged": converged,
            "sparse_fraction": np.mean(np.abs(S) > 1e-10),
            # Fitted operator: per-sample robust projection onto span(basis)
            "basis": Vt.T,
            "lambda": lam,
            "fill_values": col_means,
        }

        return L, metadata

    # ------------------------------------------------------------------
    # Baseline operator (any data)
    # ------------------------------------------------------------------

    def _robust_project(
        self,
        X: NDArray[np.floating],
        basis: NDArray[np.floating],
        lam: float,
        max_iter: int = 200,
        tol: float = 1e-8,
    ) -> NDArray[np.floating]:
        """
        Per-sample robust projection with a fixed low-rank basis.

        Minimises 0.5*||x - V t - s||^2 + lam*||s||_1 for each row, which is
        the training RPCA objective with the subspace held fixed, so the
        training baseline L is (up to the training tolerance) a fixed point.
        """
        P = basis @ basis.T
        S = np.zeros_like(X)
        for _ in range(max_iter):
            L = (X - S) @ P
            R = X - L
            S_new = np.sign(R) * np.maximum(np.abs(R) - lam, 0.0)
            if np.linalg.norm(S_new - S) <= tol * (np.linalg.norm(S) + 1e-12):
                S = S_new
                break
            S = S_new
        return (X - S) @ P

    def _apply_baseline(
        self,
        X_std: NDArray[np.floating],
        groups: Optional[NDArray[np.integer]] = None,
    ) -> NDArray[np.floating]:
        """Apply the *fitted* baseline operator to X_std."""
        meta = self.params_.baseline_metadata
        n, p = X_std.shape

        if self.baseline_type == BaselineType.MEDIAN:
            return np.tile(meta["medians"], (n, 1))

        if self.baseline_type == BaselineType.HYDROFACIES_MEDIAN:
            if groups is None:
                raise ValueError("groups required for hydrofacies baseline")
            groups = np.asarray(groups)
            X_T = np.tile(meta["global_medians"], (n, 1))
            for g, gm in meta["group_medians"].items():
                X_T[groups == g, :] = gm
            unseen = ~np.isin(groups, list(meta["group_medians"].keys()))
            if np.any(unseen):
                warnings.warn(
                    "Unseen hydrofacies groups in transform; using the global "
                    "median baseline for those samples.",
                    stacklevel=3,
                )
            return X_T

        X_work = _fill_nan_columns(X_std, meta["fill_values"])
        basis = meta["basis"]

        if self.baseline_type == BaselineType.LOW_RANK:
            return X_work @ basis @ basis.T

        if self.baseline_type == BaselineType.ROBUST_PCA:
            return self._robust_project(X_work, basis, meta["lambda"])

        raise ValueError(f"Unknown baseline type: {self.baseline_type}")

    # ------------------------------------------------------------------
    # Indeterminacy
    # ------------------------------------------------------------------

    def _indeterminacy_knn(self, X_std: NDArray[np.floating]) -> NDArray[np.floating]:
        """Local heterogeneity among the nearest training samples."""
        X_ref = self._state["X_ref"]
        fill = self._state["ref_medians"]
        k = int(max(1, min(self.n_neighbors, X_ref.shape[0])))

        tree = cKDTree(_fill_nan_columns(X_ref, fill))
        _, idx = tree.query(_fill_nan_columns(X_std, fill), k=k)
        idx = np.asarray(idx).reshape(X_std.shape[0], k)

        neighbours = X_ref[idx]  # (n, k, p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            local_var = np.nanvar(neighbours, axis=1)
        local_var = np.nan_to_num(local_var, nan=0.0)
        return 1.0 - np.exp(-local_var)

    def _indeterminacy_rolling(self, X_std: NDArray[np.floating]) -> NDArray[np.floating]:
        """Legacy rolling-window local variance over row order."""
        n, p = X_std.shape
        window = self.n_neighbors
        X_I = np.zeros_like(X_std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i in range(n):
                start = max(0, i - window // 2)
                end = min(n, i + window // 2 + 1)
                local_var = np.nanvar(X_std[start:end, :], axis=0)
                X_I[i, :] = 1.0 - np.exp(-np.nan_to_num(local_var, nan=0.0))
        return X_I

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_std: NDArray[np.floating],
        groups: Optional[NDArray[np.integer]] = None,
    ) -> "NDGEncoder":
        """
        Fit encoder to compute baseline and residual statistics.

        Parameters
        ----------
        X_std : ndarray of shape (n_samples, n_features)
            Standardized predictor matrix.
        groups : ndarray of shape (n_samples,), optional
            Group labels for hydrofacies-conditioned baseline.

        Returns
        -------
        self : NDGEncoder
            Fitted encoder.
        """
        X_std = np.asarray(X_std, dtype=np.float64)

        # Compute baseline (Truth channel template)
        if self.baseline_type == BaselineType.MEDIAN:
            X_T, metadata = self._compute_baseline_median(X_std, groups=None)
        elif self.baseline_type == BaselineType.HYDROFACIES_MEDIAN:
            if groups is None:
                raise ValueError("groups required for hydrofacies baseline")
            X_T, metadata = self._compute_baseline_median(X_std, groups=groups)
        elif self.baseline_type == BaselineType.LOW_RANK:
            X_T, metadata = self._compute_baseline_low_rank(X_std, self.baseline_rank)
        elif self.baseline_type == BaselineType.ROBUST_PCA:
            X_T, metadata = self._compute_baseline_robust_pca(X_std, self.baseline_rank)
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")

        # Compute residuals and robust scale
        R = X_std - X_T
        mad = np.nanmedian(np.abs(R), axis=0)
        sigma = 1.4826 * mad

        self.params_ = EncoderParams(
            baseline_type=self.baseline_type,
            baseline_rank=self.baseline_rank,
            falsity_map=self.falsity_map,
            falsity_params=self.falsity_params,
            sigma=sigma,
            baseline_metadata=metadata,
        )
        self._state = {
            "train_T": X_T,
            "X_ref": X_std.copy(),
            "ref_medians": np.nan_to_num(np.nanmedian(X_std, axis=0), nan=0.0),
        }
        self.is_fitted_ = True

        return self

    def transform(
        self,
        X_std: NDArray[np.floating],
        groups: Optional[NDArray[np.integer]] = None,
        indeterminacy_func: Optional[Callable] = None,
        _X_T: Optional[NDArray[np.floating]] = None,
    ) -> TripletData:
        """
        Transform standardized data to neutrosophic triplets.

        Parameters
        ----------
        X_std : ndarray of shape (n_samples, n_features)
            Standardized predictor matrix.
        groups : ndarray of shape (n_samples,), optional
            Group labels for hydrofacies baseline.
        indeterminacy_func : callable, optional
            Custom function to compute I channel. Should take X_std and
            return ndarray of shape (n_samples, n_features) in [0, 1].
            Default uses the encoder's ``indeterminacy`` mode.

        Returns
        -------
        triplets : TripletData
            Container with T, I, F channels and residuals R.
        """
        if not self.is_fitted_:
            raise RuntimeError("Encoder must be fitted before transform")

        X_std = np.asarray(X_std, dtype=np.float64)
        n, p = X_std.shape

        # Truth channel: fitted baseline operator applied to this data
        X_T = self._apply_baseline(X_std, groups) if _X_T is None else _X_T

        # Compute residuals
        R = X_std - X_T

        # Compute Falsity channel from normalized residuals
        u = np.abs(R) / (self.params_.sigma + self.delta)

        if self.falsity_map == FalsityMap.EXPONENTIAL:
            # F = 1 - exp(-u)
            X_F = 1.0 - np.exp(-u)
        elif self.falsity_map == FalsityMap.LOGISTIC:
            # F = 1 / (1 + exp(-a*(u-b)))
            a = self.falsity_params.get("a", 2.0)
            b = self.falsity_params.get("b", 1.0)
            X_F = 1.0 / (1.0 + np.exp(-a * (u - b)))
        else:
            raise ValueError(f"Unknown falsity map: {self.falsity_map}")

        # Compute Indeterminacy channel
        if indeterminacy_func is not None:
            X_I = indeterminacy_func(X_std)
        elif self.indeterminacy == IndeterminacyMode.KNN:
            X_I = self._indeterminacy_knn(X_std)
        elif self.indeterminacy == IndeterminacyMode.ROLLING:
            X_I = self._indeterminacy_rolling(X_std)
        else:
            X_I = np.zeros_like(X_std)

        # Ensure bounds [0, 1]
        X_I = np.clip(X_I, 0.0, 1.0)
        X_F = np.clip(X_F, 0.0, 1.0)

        return TripletData(T=X_T, I=X_I, F=X_F, R=R)

    def fit_transform(
        self,
        X_std: NDArray[np.floating],
        groups: Optional[NDArray[np.integer]] = None,
        indeterminacy_func: Optional[Callable] = None,
    ) -> TripletData:
        """
        Fit and transform in one step.

        The in-sample Truth channel is the training baseline itself (for
        robust PCA this is the low-rank part L of the training data;
        ``transform`` on the same data reproduces it up to the RPCA
        convergence tolerance).
        """
        self.fit(X_std, groups)
        return self.transform(
            X_std, groups, indeterminacy_func, _X_T=self._state["train_T"]
        )


def create_censoring_indeterminacy(
    X: NDArray[np.floating],
    detection_limits: NDArray[np.floating],
    iota_dl: float = 0.5,
) -> NDArray[np.floating]:
    """
    Create indeterminacy channel based on detection limit censoring.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original (non-standardized) predictor matrix.
    detection_limits : ndarray of shape (n_features,)
        Detection limits per feature.
    iota_dl : float, default=0.5
        Indeterminacy value for censored observations.

    Returns
    -------
    I : ndarray of shape (n_samples, n_features)
        Indeterminacy channel in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    detection_limits = np.asarray(detection_limits, dtype=np.float64)

    I = np.zeros_like(X)
    below_dl = X < detection_limits
    I[below_dl] = iota_dl

    return I
