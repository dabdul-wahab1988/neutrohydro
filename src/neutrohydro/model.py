"""
PNPLS (Probabilistic Neutrosophic PLS) Model for NeutroHydro.

Implements Section 4 of the specification:
- Augmented predictor space combining T, I, F channels
- Elementwise precision weights from falsity (legacy NDG encoding) or
  sample (row) weights for the Jacobian-orthogonal encoding
- PLS1 regression via NIPALS algorithm (with mean-centering and intercept)
- Optional EM-like imputation for missing data

Note on the PLS1 weight vector
------------------------------
For a single response the NIPALS weight of component h is available in
closed form, w_h = X_h^T y_h / ||X_h^T y_h|| (X_h, y_h deflated). Earlier
versions refined w_h with the update w <- X_h^T X_h w, which is a power
iteration: it converges to the leading *principal component* of X_h and
silently turned the model into principal component regression. The
closed form is used now; it matches scikit-learn's PLSRegression.
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
        Numerical tolerance used to stop extracting components when the
        remaining covariance with y (or the score norm) vanishes.
    max_iter : int, default=500
        Unused; kept for backward compatibility. PLS1 needs no inner
        iteration (see module notes).
    em_max_iter : int, default=20
        Maximum EM iterations for missing data imputation.
    em_tol : float, default=1e-4
        Convergence tolerance for EM imputation.
    center : bool, default=True
        Mean-center the (weighted) augmented predictors and the target and
        fit an intercept. Without centering, constant or non-negative
        channels (e.g. a median baseline, or I/F memberships in [0, 1])
        act as a hidden intercept and distort VIP and attribution.

    Attributes
    ----------
    components_ : PLSComponents
        Fitted PLS components and coefficients.
    x_mean_ : ndarray of shape (3p,)
        Column means of the weighted augmented matrix (zeros if center=False).
    y_mean_ : float
        Target mean (0 if center=False).
    sample_weights_ : ndarray of shape (n,) or None
        Row weights used in fitting (orthogonal encoding only).
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
        center: bool = True,
    ):
        self.n_components = n_components
        self.rho_I = rho_I
        self.rho_F = rho_F
        self.lambda_F = lambda_F
        self.tol = tol
        self.max_iter = max_iter
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.center = center

        self.components_: Optional[PLSComponents] = None
        self.x_mean_: Optional[NDArray[np.floating]] = None
        self.y_mean_: float = 0.0
        self.sample_weights_: Optional[NDArray[np.floating]] = None
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
        if getattr(triplets, "encoding", "ndg") == "orthogonal":
            # Jacobian-orthogonal channels carry signed components, not
            # memberships. Elementwise re-weighting would mix the channels
            # (it multiplies T by a function of F), so robustness is handled
            # with sample (row) weights instead -- see `fit`.
            return np.ones((triplets.T.shape[0], 3 * triplets.T.shape[1]))
        W = np.exp(-self.lambda_F * triplets.F)
        # Repeat weights for all three channels
        return np.hstack([W, W, W])

    def _weighted_design(
        self,
        triplets: TripletData,
    ) -> NDArray[np.floating]:
        """Augmented matrix after (legacy) elementwise precision weighting."""
        X_aug = self._build_augmented_matrix(triplets)
        return self._compute_precision_weights(triplets) * X_aug

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
            # PLS1 weight: closed form, proportional to covariance with y.
            # (No inner iteration: iterating w <- X'X w would converge to the
            # first principal component of X, not to the PLS direction.)
            w = X_deflated.T @ y_deflated
            w_norm = np.linalg.norm(w)
            if w_norm < self.tol:
                # No more covariance with y to explain
                k = h
                break
            w = w / w_norm

            # Score
            t = X_deflated @ w
            t_sq = t.T @ t

            if t_sq < self.tol:
                k = h
                break

            # X loading
            p_h = X_deflated.T @ t / t_sq

            # y loading
            q = y_deflated.T @ t / t_sq

            # Store
            T_scores[:, h] = t
            W_weights[:, h] = w
            P_loadings[:, h] = p_h
            q_loadings[h] = q

            # Response energy for this component
            SSY[h] = q ** 2 * t_sq

            # Deflation
            X_deflated = X_deflated - np.outer(t, p_h)
            y_deflated = y_deflated - t * q

        # Trim to actual number of components
        T_scores = T_scores[:, :k]
        W_weights = W_weights[:, :k]
        P_loadings = P_loadings[:, :k]
        q_loadings = q_loadings[:k]
        SSY = SSY[:k]

        # Compute regression coefficients: beta = W * (P'W)^-1 * q
        if k == 0:
            return PLSComponents(
                T=T_scores, W=W_weights, P=P_loadings, q=q_loadings,
                beta=np.zeros(p), n_components=0,
                explained_variance=SSY,
            )
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
        sample_weights: Optional[NDArray[np.floating]] = None,
    ) -> "PNPLS":
        """
        Fit PNPLS model on neutrosophic triplets and target.

        Parameters
        ----------
        triplets : TripletData
            Neutrosophic triplet channels from NDGEncoder or
            OrthogonalNDGEncoder.
        y : ndarray of shape (n_samples,)
            Standardized target vector.
        missing_mask : ndarray of shape (n_samples, n_features), optional
            Boolean mask where True indicates observed values.
            If provided, uses EM-like imputation.
        sample_weights : ndarray of shape (n_samples,), optional
            Row weights (weighted least squares). Defaults to
            ``triplets.sample_weights`` when the triplets come from the
            orthogonal encoder, otherwise uniform.

        Returns
        -------
        self : PNPLS
            Fitted model.
        """
        y = np.asarray(y, dtype=np.float64).ravel()

        # Elementwise-weighted augmented matrix (identity weights for the
        # orthogonal encoding)
        X_weighted = self._weighted_design(triplets)
        n = X_weighted.shape[0]

        if sample_weights is None:
            sample_weights = getattr(triplets, "sample_weights", None)
        if sample_weights is None:
            omega = np.ones(n)
        else:
            omega = np.asarray(sample_weights, dtype=np.float64).ravel()
            if omega.shape[0] != n or np.any(omega < 0) or omega.sum() <= 0:
                raise ValueError("sample_weights must be non-negative, length n")
        self.sample_weights_ = None if sample_weights is None else omega

        # Handle missing data with EM if mask provided
        if missing_mask is not None:
            X_weighted = self._em_imputation(X_weighted, y, missing_mask, omega)

        X_c, y_c = self._center(X_weighted, y, omega)
        sw = np.sqrt(omega)

        # Fit PLS via NIPALS on the row-weighted, centered problem
        self.components_ = self._nipals_pls1(
            sw[:, None] * X_c, sw * y_c, self.n_components
        )
        self.is_fitted_ = True

        return self

    def _center(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        omega: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Store (weighted) means and return centered X and y."""
        if self.center:
            wsum = omega.sum()
            self.x_mean_ = (omega[:, None] * X).sum(axis=0) / wsum
            self.y_mean_ = float((omega * y).sum() / wsum)
        else:
            self.x_mean_ = np.zeros(X.shape[1])
            self.y_mean_ = 0.0
        return X - self.x_mean_, y - self.y_mean_

    def _em_imputation(
        self,
        X_aug: NDArray[np.floating],
        y: NDArray[np.floating],
        missing_mask: NDArray[np.bool_],
        omega: Optional[NDArray[np.floating]] = None,
    ) -> NDArray[np.floating]:
        """
        EM-like imputation for missing data.

        Iteratively:
        1. M-step: Fit PLS on current completed data
        2. E-step: Impute missing values using the rank-k reconstruction
        """
        n, p_aug = X_aug.shape
        if omega is None:
            omega = np.ones(n)
        sw = np.sqrt(omega)

        # Extend mask to augmented space
        M_aug = np.hstack([missing_mask, missing_mask, missing_mask])

        X_completed = X_aug.copy()
        beta_prev = np.zeros(p_aug)

        for iteration in range(self.em_max_iter):
            # M-step: Fit PLS on current completed data
            X_c, y_c = self._center(X_completed, y, omega)
            components = self._nipals_pls1(
                sw[:, None] * X_c, sw * y_c, self.n_components
            )

            # E-step: Impute missing values using reconstruction
            # (scores were computed on row-weighted data: undo the weights)
            X_hat = (components.T @ components.P.T) / sw[:, None] + self.x_mean_

            # Update only missing entries
            X_completed = np.where(M_aug, X_aug, X_hat)

            # Check convergence
            beta_diff = np.linalg.norm(components.beta - beta_prev)
            if beta_diff < self.em_tol:
                break
            beta_prev = components.beta.copy()

        return X_completed

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

        X_weighted = self._weighted_design(triplets)
        x_mean = self.x_mean_ if self.x_mean_ is not None else 0.0

        return (X_weighted - x_mean) @ self.components_.beta + self.y_mean_

    def centered_design(
        self,
        triplets: TripletData,
    ) -> NDArray[np.floating]:
        """
        Return the weighted, centered augmented matrix used for prediction.

        ``centered_design(triplets) @ beta + y_mean_ == predict(triplets)``.
        Attribution routines use it so that contributions sum exactly to
        the predicted deviation from the intercept.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first")
        x_mean = self.x_mean_ if self.x_mean_ is not None else 0.0
        return self._weighted_design(triplets) - x_mean

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
