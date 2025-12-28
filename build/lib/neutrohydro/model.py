"""
PNPLS (Probabilistic Neutrosophic PLS) Model for NeutroHydro.

Implements Section 4 of the specification:
- Augmented predictor space combining T, I, F channels
- Elementwise precision weights from falsity
- PLS1 regression via NIPALS algorithm
- Optional EM-like imputation for missing data
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from neutrohydro.encoder import TripletData


@dataclass
class PLSComponents:
    """Container for PLS latent components and coefficients."""

    T: NDArray[np.floating]        # Latent scores (n x k)
    W: NDArray[np.floating]        # Weights (3p x k)
    P: NDArray[np.floating]        # Loadings (3p x k)
    q: NDArray[np.floating]        # Response loadings (k,)
    beta: NDArray[np.floating]     # Regression coefficients (3p,)
    n_components: int              # Number of components used
    explained_variance: NDArray[np.floating]  # Per-component explained variance


class PNPLS:
    """
    Probabilistic Neutrosophic Partial Least Squares Regression.

    Fits PLS regression on augmented predictor matrix combining the three
    neutrosophic channels (T, I, F) with channel weights and elementwise
    precision weighting from falsity.

    Parameters
    ----------
    n_components : int, default=5
        Number of PLS latent components.
    rho_I : float, default=1.0
        Weight for Indeterminacy channel in augmented space.
    rho_F : float, default=1.0
        Weight for Falsity channel in augmented space.
    lambda_F : float, default=1.0
        Strength of falsity-based precision weighting.
        Higher values downweight observations with high falsity more.
    tol : float, default=1e-6
        Convergence tolerance for NIPALS algorithm.
    max_iter : int, default=500
        Maximum iterations for NIPALS.
    em_max_iter : int, default=20
        Maximum EM iterations for missing data imputation.
    em_tol : float, default=1e-4
        Convergence tolerance for EM imputation.

    Attributes
    ----------
    components_ : PLSComponents
        Fitted PLS components and coefficients.
    is_fitted_ : bool
        Whether the model has been fitted.
    """

    def __init__(
        self,
        n_components: int = 5,
        rho_I: float = 1.0,
        rho_F: float = 1.0,
        lambda_F: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 500,
        em_max_iter: int = 20,
        em_tol: float = 1e-4,
    ):
        self.n_components = n_components
        self.rho_I = rho_I
        self.rho_F = rho_F
        self.lambda_F = lambda_F
        self.tol = tol
        self.max_iter = max_iter
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol

        self.components_: Optional[PLSComponents] = None
        self.is_fitted_: bool = False

    def _build_augmented_matrix(
        self,
        triplets: TripletData,
    ) -> NDArray[np.floating]:
        """
        Build augmented predictor matrix X^aug = [X_T, sqrt(rho_I)*X_I, sqrt(rho_F)*X_F].

        Parameters
        ----------
        triplets : TripletData
            Neutrosophic triplet channels.

        Returns
        -------
        X_aug : ndarray of shape (n_samples, 3*n_features)
            Augmented predictor matrix.
        """
        X_T = triplets.T
        X_I = np.sqrt(self.rho_I) * triplets.I
        X_F = np.sqrt(self.rho_F) * triplets.F

        return np.hstack([X_T, X_I, X_F])

    def _compute_precision_weights(
        self,
        triplets: TripletData,
    ) -> NDArray[np.floating]:
        """
        Compute elementwise precision weights from falsity.

        W_ij = exp(-lambda_F * F_ij)

        Parameters
        ----------
        triplets : TripletData
            Neutrosophic triplet channels.

        Returns
        -------
        W_aug : ndarray of shape (n_samples, 3*n_features)
            Precision weights for augmented space (repeated for each channel).
        """
        W = np.exp(-self.lambda_F * triplets.F)
        # Repeat weights for all three channels
        return np.hstack([W, W, W])

    def _nipals_pls1(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        n_components: int,
    ) -> PLSComponents:
        """
        NIPALS algorithm for PLS1 (scalar response).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Weighted augmented predictor matrix.
        y : ndarray of shape (n_samples,)
            Standardized target vector.
        n_components : int
            Number of components to extract.

        Returns
        -------
        components : PLSComponents
            Fitted PLS components.
        """
        n, p = X.shape
        k = min(n_components, min(n, p))

        # Initialize storage
        T_scores = np.zeros((n, k))
        W_weights = np.zeros((p, k))
        P_loadings = np.zeros((p, k))
        q_loadings = np.zeros(k)
        SSY = np.zeros(k)  # Response energy per component

        X_deflated = X.copy()
        y_deflated = y.copy()

        total_ss_y = np.sum(y ** 2)

        for h in range(k):
            # Initial weight: proportional to covariance with y
            w = X_deflated.T @ y_deflated
            w_norm = np.linalg.norm(w)
            if w_norm < self.tol:
                # No more variance to explain
                k = h
                break
            w = w / w_norm

            # Iterative refinement (optional, for weighted PLS)
            for _ in range(self.max_iter):
                w_old = w.copy()

                # Compute score
                t = X_deflated @ w

                # Normalize score
                t_norm = np.linalg.norm(t)
                if t_norm < self.tol:
                    break

                # Update weight
                w = X_deflated.T @ t / (t.T @ t)
                w = w / (np.linalg.norm(w) + 1e-16)

                if np.linalg.norm(w - w_old) < self.tol:
                    break

            # Final score computation
            t = X_deflated @ w
            t_sq = t.T @ t

            if t_sq < self.tol:
                k = h
                break

            # X loading
            p = X_deflated.T @ t / t_sq

            # y loading
            q = y_deflated.T @ t / t_sq

            # Store
            T_scores[:, h] = t
            W_weights[:, h] = w
            P_loadings[:, h] = p
            q_loadings[h] = q

            # Response energy for this component
            SSY[h] = q ** 2 * t_sq

            # Deflation
            X_deflated = X_deflated - np.outer(t, p)
            y_deflated = y_deflated - t * q

        # Trim to actual number of components
        T_scores = T_scores[:, :k]
        W_weights = W_weights[:, :k]
        P_loadings = P_loadings[:, :k]
        q_loadings = q_loadings[:k]
        SSY = SSY[:k]

        # Compute regression coefficients: beta = W * (P'W)^-1 * q
        PW = P_loadings.T @ W_weights
        try:
            PW_inv = np.linalg.inv(PW)
            beta = W_weights @ PW_inv @ q_loadings
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            beta = W_weights @ np.linalg.pinv(PW) @ q_loadings

        # Explained variance ratio
        explained_variance = SSY / (total_ss_y + 1e-16)

        return PLSComponents(
            T=T_scores,
            W=W_weights,
            P=P_loadings,
            q=q_loadings,
            beta=beta,
            n_components=k,
            explained_variance=explained_variance,
        )

    def fit(
        self,
        triplets: TripletData,
        y: NDArray[np.floating],
        missing_mask: Optional[NDArray[np.bool_]] = None,
    ) -> "PNPLS":
        """
        Fit PNPLS model on neutrosophic triplets and target.

        Parameters
        ----------
        triplets : TripletData
            Neutrosophic triplet channels from NDGEncoder.
        y : ndarray of shape (n_samples,)
            Standardized target vector.
        missing_mask : ndarray of shape (n_samples, n_features), optional
            Boolean mask where True indicates observed values.
            If provided, uses EM-like imputation.

        Returns
        -------
        self : PNPLS
            Fitted model.
        """
        y = np.asarray(y, dtype=np.float64).ravel()

        # Build augmented matrix
        X_aug = self._build_augmented_matrix(triplets)

        # Compute precision weights
        W_precision = self._compute_precision_weights(triplets)

        # Apply elementwise weighting
        X_weighted = W_precision * X_aug

        # Handle missing data with EM if mask provided
        if missing_mask is not None:
            X_weighted, y = self._em_imputation(X_weighted, y, missing_mask, triplets)

        # Fit PLS via NIPALS
        self.components_ = self._nipals_pls1(X_weighted, y, self.n_components)
        self.is_fitted_ = True

        return self

    def _em_imputation(
        self,
        X_aug: NDArray[np.floating],
        y: NDArray[np.floating],
        missing_mask: NDArray[np.bool_],
        triplets: TripletData,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        EM-like imputation for missing data.

        Iteratively:
        1. E-step: Impute missing values using current low-rank reconstruction
        2. M-step: Refit PLS on completed data
        """
        n, p_orig = triplets.T.shape
        p_aug = 3 * p_orig

        # Extend mask to augmented space
        M_aug = np.hstack([missing_mask, missing_mask, missing_mask])

        X_completed = X_aug.copy()
        beta_prev = np.zeros(p_aug)

        for iteration in range(self.em_max_iter):
            # M-step: Fit PLS on current completed data
            components = self._nipals_pls1(X_completed, y, self.n_components)

            # E-step: Impute missing values using reconstruction
            X_hat = components.T @ components.P.T

            # Update only missing entries
            X_completed = np.where(M_aug, X_aug, X_hat)

            # Check convergence
            beta_diff = np.linalg.norm(components.beta - beta_prev)
            if beta_diff < self.em_tol:
                break
            beta_prev = components.beta.copy()

        return X_completed, y

    def predict(
        self,
        triplets: TripletData,
    ) -> NDArray[np.floating]:
        """
        Predict target values for new triplet data.

        Parameters
        ----------
        triplets : TripletData
            Neutrosophic triplet channels.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values (standardized scale).
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict")

        X_aug = self._build_augmented_matrix(triplets)
        W_precision = self._compute_precision_weights(triplets)
        X_weighted = W_precision * X_aug

        return X_weighted @ self.components_.beta

    def get_coefficients(self) -> dict[str, NDArray[np.floating]]:
        """
        Get regression coefficients partitioned by channel.

        Returns
        -------
        coefficients : dict
            Dictionary with keys 'beta_T', 'beta_I', 'beta_F' containing
            coefficients for each channel.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first")

        beta = self.components_.beta
        p = len(beta) // 3

        return {
            "beta_T": beta[:p],
            "beta_I": beta[p:2*p],
            "beta_F": beta[2*p:],
        }

    def get_weights_by_channel(self) -> dict[str, NDArray[np.floating]]:
        """
        Get PLS weights partitioned by channel.

        Returns
        -------
        weights : dict
            Dictionary with keys 'W_T', 'W_I', 'W_F' containing
            weight matrices (p x k) for each channel.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first")

        W = self.components_.W
        p = W.shape[0] // 3

        return {
            "W_T": W[:p, :],
            "W_I": W[p:2*p, :],
            "W_F": W[2*p:, :],
        }

    def score(
        self,
        triplets: TripletData,
        y: NDArray[np.floating],
    ) -> float:
        """
        Compute R^2 score for the model.

        Parameters
        ----------
        triplets : TripletData
            Neutrosophic triplet channels.
        y : ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            Coefficient of determination.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(triplets)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1.0 - ss_res / (ss_tot + 1e-16)

    def get_components(self) -> PLSComponents:
        """Return fitted PLS components."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first")
        return self.components_
