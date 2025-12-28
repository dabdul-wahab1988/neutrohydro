"""
NDG Encoder (Neutrosophic Data Generator) for NeutroHydro.

Implements Section 3 of the specification:
- Maps scalar values to neutrosophic triplets (T, I, F)
- T (Truth): Baseline component via robust baseline operators
- I (Indeterminacy): Uncertainty/ambiguity channel
- F (Falsity): Perturbation likelihood from residuals
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd


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
    """Container for neutrosophic triplet channels."""

    T: NDArray[np.floating]  # Truth channel (baseline)
    I: NDArray[np.floating]  # Indeterminacy channel (ambiguity)
    F: NDArray[np.floating]  # Falsity channel (perturbation)
    R: NDArray[np.floating]  # Residuals (X_std - T)


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

    Attributes
    ----------
    params_ : EncoderParams
        Fitted encoder parameters.
    is_fitted_ : bool
        Whether the encoder has been fitted.
    """

    def __init__(
        self,
        baseline_type: BaselineType | str = BaselineType.MEDIAN,
        baseline_rank: Optional[int] = None,
        falsity_map: FalsityMap | str = FalsityMap.EXPONENTIAL,
        falsity_params: Optional[dict] = None,
        delta: float = 1e-10,
    ):
        if isinstance(baseline_type, str):
            baseline_type = BaselineType(baseline_type)
        if isinstance(falsity_map, str):
            falsity_map = FalsityMap(falsity_map)

        self.baseline_type = baseline_type
        self.baseline_rank = baseline_rank
        self.falsity_map = falsity_map
        self.delta = delta

        # Set default falsity parameters
        if falsity_params is None:
            if falsity_map == FalsityMap.LOGISTIC:
                falsity_params = {"a": 2.0, "b": 1.0}
            else:
                falsity_params = {}
        self.falsity_params = falsity_params

        self.params_: Optional[EncoderParams] = None
        self.is_fitted_: bool = False

        # Validate rank requirement
        if baseline_type in [BaselineType.LOW_RANK, BaselineType.ROBUST_PCA]:
            if baseline_rank is None:
                raise ValueError(
                    f"baseline_rank required for {baseline_type.value}"
                )

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

            metadata = {"type": "hydrofacies_median", "group_medians": group_medians}

        return X_T, metadata

    def _compute_baseline_low_rank(
        self,
        X_std: NDArray[np.floating],
        rank: int,
    ) -> tuple[NDArray[np.floating], dict]:
        """Compute low-rank baseline via truncated SVD."""
        # Handle missing values with mean imputation for SVD
        X_work = X_std.copy()
        nan_mask = np.isnan(X_work)
        if np.any(nan_mask):
            col_means = np.nanmean(X_work, axis=0)
            for j in range(X_work.shape[1]):
                X_work[nan_mask[:, j], j] = col_means[j]

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
        }

        return X_T, metadata

    def _compute_baseline_robust_pca(
        self,
        X_std: NDArray[np.floating],
        rank: int,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[NDArray[np.floating], dict]:
        """
        Compute robust PCA baseline (L + S decomposition).

        Uses iterative thresholding: X = L + S where L is low-rank and S is sparse.
        """
        X_work = X_std.copy()
        nan_mask = np.isnan(X_work)
        if np.any(nan_mask):
            col_means = np.nanmean(X_work, axis=0)
            for j in range(X_work.shape[1]):
                X_work[nan_mask[:, j], j] = col_means[j]

        n, p = X_work.shape
        lam = 1.0 / np.sqrt(max(n, p))  # Regularization parameter

        L = np.zeros_like(X_work)
        S = np.zeros_like(X_work)

        for iteration in range(max_iter):
            L_old = L.copy()

            # Update L (low-rank via SVD thresholding)
            U, s, Vt = svd(X_work - S, full_matrices=False)
            r = min(rank, len(s))
            L = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]

            # Update S (sparse via soft thresholding)
            residual = X_work - L
            S = np.sign(residual) * np.maximum(np.abs(residual) - lam, 0)

            # Check convergence
            if np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-10) < tol:
                break

        metadata = {
            "type": "robust_pca",
            "rank": r,
            "iterations": iteration + 1,
            "sparse_fraction": np.mean(np.abs(S) > 1e-10),
        }

        return L, metadata

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
        self.is_fitted_ = True

        return self

    def transform(
        self,
        X_std: NDArray[np.floating],
        groups: Optional[NDArray[np.integer]] = None,
        indeterminacy_func: Optional[Callable] = None,
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
            Default uses uniform small indeterminacy.

        Returns
        -------
        triplets : TripletData
            Container with T, I, F channels and residuals R.
        """
        if not self.is_fitted_:
            raise RuntimeError("Encoder must be fitted before transform")

        X_std = np.asarray(X_std, dtype=np.float64)
        n, p = X_std.shape

        # Compute Truth channel (baseline)
        if self.baseline_type == BaselineType.MEDIAN:
            X_T, _ = self._compute_baseline_median(X_std, groups=None)
        elif self.baseline_type == BaselineType.HYDROFACIES_MEDIAN:
            X_T, _ = self._compute_baseline_median(X_std, groups=groups)
        elif self.baseline_type == BaselineType.LOW_RANK:
            X_T, _ = self._compute_baseline_low_rank(X_std, self.baseline_rank)
        elif self.baseline_type == BaselineType.ROBUST_PCA:
            X_T, _ = self._compute_baseline_robust_pca(X_std, self.baseline_rank)

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
        else:
            # Default: small uniform indeterminacy based on local variance
            # Using a simplified local heterogeneity measure
            X_I = np.zeros_like(X_std)
            for j in range(p):
                col = X_std[:, j]
                # Simple local variance (rolling window approximation)
                window = 5
                for i in range(n):
                    start = max(0, i - window // 2)
                    end = min(n, i + window // 2 + 1)
                    local_var = np.nanvar(col[start:end])
                    # Map to [0, 1] via saturation
                    X_I[i, j] = 1.0 - np.exp(-local_var)

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
        """Fit and transform in one step."""
        self.fit(X_std, groups)
        return self.transform(X_std, groups, indeterminacy_func)


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
