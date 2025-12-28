"""
Preprocessing module for NeutroHydro.

Implements Section 2 of the specification:
- Positivity and log transforms
- Robust centering and scaling (median/MAD)
- Handling of missing data and detection limits
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class PreprocessorParams:
    """Stores preprocessing parameters for reproducible transforms."""

    mu: NDArray[np.floating]  # Robust centers (median)
    s: NDArray[np.floating]   # Robust scales (MAD)
    mu_y: float               # Target center
    s_y: float                # Target scale
    log_transform: bool       # Whether log transform was applied
    delta_x: float            # Small constant for log stability
    delta_s: float            # Small constant for scale stability
    feature_names: Optional[list[str]] = None


class Preprocessor:
    """
    Non-compositional preprocessor for groundwater ion data.

    Applies robust centering (median) and scaling (MAD) to predictor
    matrix X and target y. Optionally applies log transform for
    data spanning orders of magnitude.

    Parameters
    ----------
    log_transform : bool, default=False
        If True, apply log(X + delta_x) before centering/scaling.
    delta_x : float, default=1e-12
        Small constant added before log transform for numerical stability.
    delta_s : float, default=1e-10
        Small constant added to scale to prevent division by zero.

    Attributes
    ----------
    params_ : PreprocessorParams
        Fitted preprocessing parameters.
    is_fitted_ : bool
        Whether the preprocessor has been fitted.
    """

    def __init__(
        self,
        log_transform: bool = False,
        delta_x: float = 1e-12,
        delta_s: float = 1e-10,
    ):
        self.log_transform = log_transform
        self.delta_x = delta_x
        self.delta_s = delta_s
        self.params_: Optional[PreprocessorParams] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        feature_names: Optional[list[str]] = None,
    ) -> "Preprocessor":
        """
        Fit preprocessor to compute robust centers and scales.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix with ion concentrations (non-negative).
        y : ndarray of shape (n_samples,)
            Target vector (e.g., log TDS, log EC).
        feature_names : list of str, optional
            Names of ion features for reference.

        Returns
        -------
        self : Preprocessor
            Fitted preprocessor.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)}")

        # Apply log transform if requested
        if self.log_transform:
            X_work = np.log(X + self.delta_x)
        else:
            X_work = X.copy()

        # Compute robust centers (median) for each feature
        mu = np.nanmedian(X_work, axis=0)

        # Compute robust scales (MAD) for each feature
        # MAD = median(|X - median(X)|)
        # Scaled by 1.4826 to be consistent with std for normal distributions
        deviations = np.abs(X_work - mu)
        mad = np.nanmedian(deviations, axis=0)
        s = 1.4826 * mad

        # Compute target center and scale
        mu_y = float(np.nanmedian(y))
        mad_y = float(np.nanmedian(np.abs(y - mu_y)))
        s_y = 1.4826 * mad_y

        self.params_ = PreprocessorParams(
            mu=mu,
            s=s,
            mu_y=mu_y,
            s_y=s_y,
            log_transform=self.log_transform,
            delta_x=self.delta_x,
            delta_s=self.delta_s,
            feature_names=feature_names,
        )
        self.is_fitted_ = True

        return self

    def transform(
        self,
        X: NDArray[np.floating],
        y: Optional[NDArray[np.floating]] = None,
    ) -> tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
        """
        Transform X (and optionally y) using fitted parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix.
        y : ndarray of shape (n_samples,), optional
            Target vector.

        Returns
        -------
        X_std : ndarray of shape (n_samples, n_features)
            Standardized predictor matrix.
        y_std : ndarray of shape (n_samples,) or None
            Standardized target vector if y was provided.
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform")

        X = np.asarray(X, dtype=np.float64)

        # Apply log transform if used during fitting
        if self.params_.log_transform:
            X_work = np.log(X + self.params_.delta_x)
        else:
            X_work = X.copy()

        # Standardize: (X - mu) / (s + delta_s)
        X_std = (X_work - self.params_.mu) / (self.params_.s + self.params_.delta_s)

        y_std = None
        if y is not None:
            y = np.asarray(y, dtype=np.float64).ravel()
            y_std = (y - self.params_.mu_y) / (self.params_.s_y + self.params_.delta_s)

        return X_std, y_std

    def fit_transform(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        feature_names: Optional[list[str]] = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix.
        y : ndarray of shape (n_samples,)
            Target vector.
        feature_names : list of str, optional
            Names of ion features.

        Returns
        -------
        X_std : ndarray of shape (n_samples, n_features)
            Standardized predictor matrix.
        y_std : ndarray of shape (n_samples,)
            Standardized target vector.
        """
        self.fit(X, y, feature_names)
        X_std, y_std = self.transform(X, y)
        return X_std, y_std

    def inverse_transform_y(
        self,
        y_std: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Inverse transform standardized target back to original scale.

        Parameters
        ----------
        y_std : ndarray
            Standardized target values.

        Returns
        -------
        y : ndarray
            Target values in original scale.
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")

        y_std = np.asarray(y_std, dtype=np.float64)
        return y_std * (self.params_.s_y + self.params_.delta_s) + self.params_.mu_y

    def get_params(self) -> PreprocessorParams:
        """Return fitted preprocessing parameters."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted first")
        return self.params_


def handle_missing_data(
    X: NDArray[np.floating],
    detection_limits: Optional[NDArray[np.floating]] = None,
    fill_method: str = "median",
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Handle missing data and detection limits in predictor matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Predictor matrix with potential NaN values.
    detection_limits : ndarray of shape (n_features,), optional
        Detection limits per feature. Values below DL are treated as censored.
    fill_method : str, default="median"
        Method for filling missing values: "median", "zero", or "dl_half"
        (half of detection limit).

    Returns
    -------
    X_filled : ndarray of shape (n_samples, n_features)
        Matrix with missing values filled.
    M : ndarray of shape (n_samples, n_features)
        Boolean mask where True indicates observed (non-missing) values.
    """
    X = np.asarray(X, dtype=np.float64)

    # Create missingness mask (True = observed)
    M = ~np.isnan(X)

    # Handle detection limits
    if detection_limits is not None:
        detection_limits = np.asarray(detection_limits, dtype=np.float64)
        below_dl = X < detection_limits
        # Mark values below DL as censored (part of missing mask)
        M = M & ~below_dl

    # Fill missing values
    X_filled = X.copy()

    if fill_method == "median":
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            missing_mask = ~M[:, j]
            X_filled[missing_mask, j] = col_medians[j]

    elif fill_method == "zero":
        X_filled[~M] = 0.0

    elif fill_method == "dl_half":
        if detection_limits is None:
            raise ValueError("detection_limits required for dl_half method")
        for j in range(X.shape[1]):
            missing_mask = ~M[:, j]
            X_filled[missing_mask, j] = detection_limits[j] / 2.0

    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    return X_filled, M
