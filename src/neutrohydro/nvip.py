"""
NVIP (Neutrosophic Variable Importance in Projection) for NeutroHydro.

Implements Section 5 of the specification:
- Channel-wise VIP decomposition (VIP_T, VIP_I, VIP_F)
- L2 decomposition theorem: VIP_agg^2 = VIP_T^2 + VIP_I^2 + VIP_F^2
- Component energy calculations
"""

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from numpy.typing import NDArray

from neutrohydro.model import PNPLS, PLSComponents


@dataclass
class NVIPResult:
    """Container for NVIP computation results."""

    VIP_T: NDArray[np.floating]    # Truth channel VIP per variable
    VIP_I: NDArray[np.floating]    # Indeterminacy channel VIP per variable
    VIP_F: NDArray[np.floating]    # Falsity channel VIP per variable
    VIP_agg: NDArray[np.floating]  # Aggregated VIP per variable
    E_T: NDArray[np.floating]      # Truth energy (VIP_T^2)
    E_I: NDArray[np.floating]      # Indeterminacy energy (VIP_I^2)
    E_F: NDArray[np.floating]      # Falsity energy (VIP_F^2)
    SSY: NDArray[np.floating]      # Response energy per component
    n_features: int                # Number of original features (p)
    n_components: int              # Number of PLS components (k)
    model: Optional[Any] = None    # Reference to fitted model
    feature_names: Optional[list[str]] = None # Feature names


def compute_nvip(
    model: PNPLS,
    components: Optional[PLSComponents] = None,
    feature_names: Optional[list[str]] = None,
) -> NVIPResult:
    """
    Compute Neutrosophic Variable Importance in Projection.

    Decomposes variable importance across Truth, Indeterminacy, and Falsity
    channels. Implements the L2 decomposition theorem:

        VIP_agg^2(j) = VIP_T^2(j) + VIP_I^2(j) + VIP_F^2(j)

    Parameters
    ----------
    model : PNPLS
        Fitted PNPLS model.
    components : PLSComponents, optional
        Pre-computed PLS components. If None, retrieved from model.

    Returns
    -------
    result : NVIPResult
        Container with channel-wise VIPs and energies.

    Notes
    -----
    The VIP formula for channel c and variable j is:

        VIP_c(j) = sqrt(p * sum_h(SSY_h * omega_c,h(j) / Omega_h) / sum_h(SSY_h))

    where:
        - omega_c,h(j) = w_c,h(j)^2 (squared weight for channel c, component h)
        - Omega_h = ||w_h||^2 (total squared weight for component h)
        - SSY_h = q_h^2 * (t_h' t_h) (response energy for component h)
    """
    if not model.is_fitted_:
        raise RuntimeError("Model must be fitted before computing NVIP")

    if components is None:
        components = model.get_components()

    W = components.W       # (3p x k) weight matrix
    T = components.T       # (n x k) score matrix
    q = components.q       # (k,) response loadings

    p3 = W.shape[0]
    p = p3 // 3            # Original number of features
    k = components.n_components

    # Partition weights by channel
    W_T = W[:p, :]         # Truth weights
    W_I = W[p:2*p, :]      # Indeterminacy weights
    W_F = W[2*p:, :]       # Falsity weights

    # Compute squared weights per variable per component
    omega_T = W_T ** 2     # (p x k)
    omega_I = W_I ** 2     # (p x k)
    omega_F = W_F ** 2     # (p x k)

    # Total squared weight per component (normalization)
    Omega = np.sum(W ** 2, axis=0)  # (k,)

    # Response energy per component: SSY_h = q_h^2 * (t_h' t_h)
    t_sq = np.sum(T ** 2, axis=0)   # (k,)
    SSY = q ** 2 * t_sq             # (k,)

    # Total response energy
    total_SSY = np.sum(SSY)
    if total_SSY < 1e-16:
        # No explained variance, return zeros
        return NVIPResult(
            VIP_T=np.zeros(p),
            VIP_I=np.zeros(p),
            VIP_F=np.zeros(p),
            VIP_agg=np.zeros(p),
            E_T=np.zeros(p),
            E_I=np.zeros(p),
            E_F=np.zeros(p),
            SSY=SSY,
            n_features=p,
            n_components=k,
        )

    # Compute channel-wise VIP energies
    # E_c(j) = p * sum_h(SSY_h * omega_c,h(j) / Omega_h) / sum_h(SSY_h)

    # Weighted ratio: SSY_h / Omega_h
    weight_ratio = SSY / (Omega + 1e-16)  # (k,)

    # Sum over components for each variable
    E_T = p * (omega_T @ weight_ratio) / total_SSY  # (p,)
    E_I = p * (omega_I @ weight_ratio) / total_SSY  # (p,)
    E_F = p * (omega_F @ weight_ratio) / total_SSY  # (p,)

    # VIPs are square roots of energies
    VIP_T = np.sqrt(np.maximum(E_T, 0))
    VIP_I = np.sqrt(np.maximum(E_I, 0))
    VIP_F = np.sqrt(np.maximum(E_F, 0))

    # Aggregated VIP (L2 decomposition theorem)
    E_agg = E_T + E_I + E_F
    VIP_agg = np.sqrt(np.maximum(E_agg, 0))

    return NVIPResult(
        VIP_T=VIP_T,
        VIP_I=VIP_I,
        VIP_F=VIP_F,
        VIP_agg=VIP_agg,
        E_T=E_T,
        E_I=E_I,
        E_F=E_F,
        SSY=SSY,
        n_features=p,
        n_components=k,
        model=model,
        feature_names=feature_names
    )


def verify_l2_decomposition(nvip_result: NVIPResult, tol: float = 1e-10) -> bool:
    """
    Verify the NVIP L2 decomposition theorem.

    Checks that VIP_agg^2(j) = VIP_T^2(j) + VIP_I^2(j) + VIP_F^2(j)
    for all variables j.

    Parameters
    ----------
    nvip_result : NVIPResult
        NVIP computation result.
    tol : float, default=1e-10
        Tolerance for numerical comparison.

    Returns
    -------
    valid : bool
        True if decomposition holds within tolerance.
    """
    lhs = nvip_result.VIP_agg ** 2
    rhs = nvip_result.VIP_T ** 2 + nvip_result.VIP_I ** 2 + nvip_result.VIP_F ** 2

    return np.allclose(lhs, rhs, atol=tol)


def compute_nvip_bootstrap(
    model: PNPLS,
    triplets,
    y: NDArray[np.floating],
    n_bootstrap: int = 100,
    random_state: Optional[int] = None,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute bootstrap confidence intervals for NVIP.

    Parameters
    ----------
    model : PNPLS
        PNPLS model (will be refit for each bootstrap sample).
    triplets : TripletData
        Original triplet data.
    y : ndarray of shape (n_samples,)
        Target vector.
    n_bootstrap : int, default=100
        Number of bootstrap samples.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    bootstrap_results : dict
        Dictionary with keys:
        - 'VIP_T_mean', 'VIP_T_std', 'VIP_T_ci_lower', 'VIP_T_ci_upper'
        - 'VIP_I_mean', 'VIP_I_std', 'VIP_I_ci_lower', 'VIP_I_ci_upper'
        - 'VIP_F_mean', 'VIP_F_std', 'VIP_F_ci_lower', 'VIP_F_ci_upper'
        - 'VIP_agg_mean', 'VIP_agg_std', 'VIP_agg_ci_lower', 'VIP_agg_ci_upper'
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    p = triplets.T.shape[1]

    # Storage for bootstrap samples
    VIP_T_samples = np.zeros((n_bootstrap, p))
    VIP_I_samples = np.zeros((n_bootstrap, p))
    VIP_F_samples = np.zeros((n_bootstrap, p))
    VIP_agg_samples = np.zeros((n_bootstrap, p))

    from neutrohydro.encoder import TripletData

    for b in range(n_bootstrap):
        # Bootstrap resample
        idx = rng.choice(n, size=n, replace=True)

        # Create resampled triplets
        triplets_b = TripletData(
            T=triplets.T[idx, :],
            I=triplets.I[idx, :],
            F=triplets.F[idx, :],
            R=triplets.R[idx, :],
        )
        y_b = y[idx]

        # Fit model on bootstrap sample
        model_b = PNPLS(
            n_components=model.n_components,
            rho_I=model.rho_I,
            rho_F=model.rho_F,
            lambda_F=model.lambda_F,
        )
        model_b.fit(triplets_b, y_b)

        # Compute NVIP
        nvip_b = compute_nvip(model_b)

        VIP_T_samples[b, :] = nvip_b.VIP_T
        VIP_I_samples[b, :] = nvip_b.VIP_I
        VIP_F_samples[b, :] = nvip_b.VIP_F
        VIP_agg_samples[b, :] = nvip_b.VIP_agg

    # Compute statistics
    def compute_stats(samples):
        return {
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'ci_lower': np.percentile(samples, 2.5, axis=0),
            'ci_upper': np.percentile(samples, 97.5, axis=0),
        }

    results = {}
    for name, samples in [
        ('VIP_T', VIP_T_samples),
        ('VIP_I', VIP_I_samples),
        ('VIP_F', VIP_F_samples),
        ('VIP_agg', VIP_agg_samples),
    ]:
        stats = compute_stats(samples)
        results[f'{name}_mean'] = stats['mean']
        results[f'{name}_std'] = stats['std']
        results[f'{name}_ci_lower'] = stats['ci_lower']
        results[f'{name}_ci_upper'] = stats['ci_upper']

    return results


def nvip_to_dataframe(
    nvip_result: NVIPResult,
    feature_names: Optional[list[str]] = None,
):
    """
    Convert NVIP result to pandas DataFrame.

    Parameters
    ----------
    nvip_result : NVIPResult
        NVIP computation result.
    feature_names : list of str, optional
        Names for features. If None, uses indices.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns: feature, VIP_T, VIP_I, VIP_F, VIP_agg,
        E_T, E_I, E_F.
    """
    import pandas as pd

    p = nvip_result.n_features
    if feature_names is None:
        feature_names = [f"X{j}" for j in range(p)]

    return pd.DataFrame({
        'feature': feature_names,
        'VIP_T': nvip_result.VIP_T,
        'VIP_I': nvip_result.VIP_I,
        'VIP_F': nvip_result.VIP_F,
        'VIP_agg': nvip_result.VIP_agg,
        'E_T': nvip_result.E_T,
        'E_I': nvip_result.E_I,
        'E_F': nvip_result.E_F,
    })
