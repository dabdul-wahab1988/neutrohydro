"""
Attribution metrics for NeutroHydro.

Implements Sections 6-7 of the specification:
- NSR (Neutrosophic Source Ratio): baseline vs perturbation odds
- pi_G: baseline fraction per ion
- G_i: sample-level baseline fraction of prediction
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from neutrohydro.nvip import NVIPResult
from neutrohydro.model import PNPLS
from neutrohydro.encoder import TripletData


@dataclass
class NSRResult:
    """Container for NSR and baseline fraction results."""

    NSR: NDArray[np.floating]        # Neutrosophic Source Ratio per ion
    pi_G: NDArray[np.floating]       # Baseline fraction per ion [0,1]
    pi_A: NDArray[np.floating]       # Perturbation fraction per ion [0,1]
    E_T: NDArray[np.floating]        # Truth energy per ion
    E_P: NDArray[np.floating]        # Perturbation energy per ion (E_I + E_F)
    classification: NDArray[np.str_] # Classification: "baseline", "perturbation", or "mixed"
    gamma: float                     # Classification threshold used


def compute_nsr(
    nvip_result: NVIPResult,
    epsilon: float = 1e-10,
    gamma: float = 0.7,
) -> NSRResult:
    """
    Compute Neutrosophic Source Ratio and baseline fractions.

    Parameters
    ----------
    nvip_result : NVIPResult
        NVIP computation result containing channel-wise VIPs.
    epsilon : float, default=1e-10
        Small constant for numerical stability in ratio computation.
    gamma : float, default=0.7
        Classification threshold. Ion is:
        - baseline-dominant if pi_G >= gamma
        - perturbation-dominant if pi_G <= 1 - gamma
        - mixed otherwise

    Returns
    -------
    result : NSRResult
        Container with NSR, baseline fractions, and classifications.

    Notes
    -----
    Energies and ratios:
        E_T(j) = VIP_T^2(j)
        E_P(j) = VIP_I^2(j) + VIP_F^2(j)

        NSR(j) = (E_T(j) + epsilon) / (E_P(j) + epsilon)
        pi_G(j) = E_T(j) / (E_T(j) + E_P(j))

    Relationship:
        pi_G(j) = NSR(j) / (1 + NSR(j))
        NSR(j) = pi_G(j) / (1 - pi_G(j))
    """
    # Extract energies from NVIP
    E_T = nvip_result.E_T                          # VIP_T^2
    E_P = nvip_result.E_I + nvip_result.E_F        # VIP_I^2 + VIP_F^2

    # Compute NSR (odds ratio)
    NSR = (E_T + epsilon) / (E_P + epsilon)

    # Compute baseline fraction
    total_energy = E_T + E_P
    pi_G = np.where(total_energy > epsilon, E_T / total_energy, 0.5)
    pi_G = np.clip(pi_G, 0.0, 1.0)

    # Perturbation fraction
    pi_A = 1.0 - pi_G

    # Classification
    p = len(pi_G)
    classification = np.empty(p, dtype='U12')
    classification[:] = "mixed"
    classification[pi_G >= gamma] = "baseline"
    classification[pi_G <= 1 - gamma] = "perturbation"

    return NSRResult(
        NSR=NSR,
        pi_G=pi_G,
        pi_A=pi_A,
        E_T=E_T,
        E_P=E_P,
        classification=classification,
        gamma=gamma,
    )


@dataclass
class SampleAttributionResult:
    """Container for sample-level attribution results."""

    G: NDArray[np.floating]          # Baseline fraction per sample [0,1]
    A: NDArray[np.floating]          # Perturbation fraction per sample [0,1]
    w: NDArray[np.floating]          # Attribution mass per sample per ion (n x p)
    c: NDArray[np.floating]          # Net contribution per sample per ion (n x p)


def compute_sample_baseline_fraction(
    model: PNPLS,
    triplets: TripletData,
    nsr_result: NSRResult,
) -> SampleAttributionResult:
    """
    Compute sample-level baseline fraction of prediction (G_i).

    Parameters
    ----------
    model : PNPLS
        Fitted PNPLS model.
    triplets : TripletData
        Neutrosophic triplet channels.
    nsr_result : NSRResult
        NSR computation result containing pi_G per ion.

    Returns
    -------
    result : SampleAttributionResult
        Container with sample-level G_i and per-ion contributions.

    Notes
    -----
    For sample i:
        c_ij = (X_T)_ij * beta_T(j) + (X_I)_ij * beta_I(j) + (X_F)_ij * beta_F(j)
        w_ij = |c_ij|  (attribution mass)

        G_i = sum_j(pi_G(j) * w_ij) / sum_j(w_ij)

    Interpretation:
        G_i is the fraction of the model's absolute predictive attribution
        mass carried by baseline-dominant ions.
    """
    if not model.is_fitted_:
        raise RuntimeError("Model must be fitted")

    # Get coefficients partitioned by channel
    coeffs = model.get_coefficients()
    beta_T = coeffs["beta_T"]
    beta_I = coeffs["beta_I"]
    beta_F = coeffs["beta_F"]

    # Get precision-weighted data
    W_precision = np.exp(-model.lambda_F * triplets.F)

    # Weighted channels
    X_T_weighted = W_precision * triplets.T
    X_I_weighted = W_precision * np.sqrt(model.rho_I) * triplets.I
    X_F_weighted = W_precision * np.sqrt(model.rho_F) * triplets.F

    # Compute net contribution per ion per sample
    # c_ij = weighted_T * beta_T + weighted_I * beta_I + weighted_F * beta_F
    c = X_T_weighted * beta_T + X_I_weighted * beta_I + X_F_weighted * beta_F

    # Attribution mass (absolute value to avoid sign cancellation)
    w = np.abs(c)

    # Sample-level baseline fraction
    # G_i = sum_j(pi_G(j) * w_ij) / sum_j(w_ij)
    pi_G = nsr_result.pi_G

    numerator = w @ pi_G          # (n,)
    denominator = w.sum(axis=1)   # (n,)

    G = np.where(denominator > 1e-16, numerator / denominator, 0.5)
    G = np.clip(G, 0.0, 1.0)

    A = 1.0 - G

    return SampleAttributionResult(
        G=G,
        A=A,
        w=w,
        c=c,
    )


def attribution_summary(
    nsr_result: NSRResult,
    sample_result: SampleAttributionResult,
    feature_names: Optional[list[str]] = None,
) -> dict:
    """
    Generate summary statistics for attribution analysis.

    Parameters
    ----------
    nsr_result : NSRResult
        NSR results for ions.
    sample_result : SampleAttributionResult
        Sample-level attribution results.
    feature_names : list of str, optional
        Names of ions/features.

    Returns
    -------
    summary : dict
        Dictionary with summary statistics.
    """
    p = len(nsr_result.pi_G)
    n = len(sample_result.G)

    if feature_names is None:
        feature_names = [f"Ion_{j}" for j in range(p)]

    # Ion-level summary
    n_baseline = np.sum(nsr_result.classification == "baseline")
    n_perturbation = np.sum(nsr_result.classification == "perturbation")
    n_mixed = np.sum(nsr_result.classification == "mixed")

    # Find most baseline-dominant and perturbation-dominant ions
    baseline_ions = [feature_names[j] for j in np.where(nsr_result.classification == "baseline")[0]]
    perturbation_ions = [feature_names[j] for j in np.where(nsr_result.classification == "perturbation")[0]]

    # Sample-level summary
    G = sample_result.G
    G_mean = float(np.mean(G))
    G_std = float(np.std(G))
    G_median = float(np.median(G))

    # Fraction of samples baseline-dominant
    frac_baseline_samples = float(np.mean(G >= nsr_result.gamma))
    frac_perturbation_samples = float(np.mean(G <= 1 - nsr_result.gamma))

    return {
        # Ion-level
        "n_ions": p,
        "n_baseline_ions": int(n_baseline),
        "n_perturbation_ions": int(n_perturbation),
        "n_mixed_ions": int(n_mixed),
        "baseline_ions": baseline_ions,
        "perturbation_ions": perturbation_ions,
        "mean_pi_G": float(np.mean(nsr_result.pi_G)),
        "mean_NSR": float(np.mean(nsr_result.NSR)),
        # Sample-level
        "n_samples": n,
        "G_mean": G_mean,
        "G_std": G_std,
        "G_median": G_median,
        "frac_baseline_samples": frac_baseline_samples,
        "frac_perturbation_samples": frac_perturbation_samples,
        "classification_threshold": nsr_result.gamma,
    }


def nsr_to_dataframe(
    nsr_result: NSRResult,
    feature_names: Optional[list[str]] = None,
):
    """
    Convert NSR result to pandas DataFrame.

    Parameters
    ----------
    nsr_result : NSRResult
        NSR computation result.
    feature_names : list of str, optional
        Names for ions/features.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with ion-level attribution metrics.
    """
    import pandas as pd

    p = len(nsr_result.pi_G)
    if feature_names is None:
        feature_names = [f"Ion_{j}" for j in range(p)]

    return pd.DataFrame({
        'ion': feature_names,
        'NSR': nsr_result.NSR,
        'pi_G': nsr_result.pi_G,
        'pi_A': nsr_result.pi_A,
        'E_T': nsr_result.E_T,
        'E_P': nsr_result.E_P,
        'classification': nsr_result.classification,
    })


def sample_attribution_to_dataframe(
    sample_result: SampleAttributionResult,
    sample_ids: Optional[list] = None,
):
    """
    Convert sample attribution result to pandas DataFrame.

    Parameters
    ----------
    sample_result : SampleAttributionResult
        Sample-level attribution result.
    sample_ids : list, optional
        Sample identifiers.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with sample-level attribution metrics.
    """
    import pandas as pd

    n = len(sample_result.G)
    if sample_ids is None:
        sample_ids = list(range(n))

    return pd.DataFrame({
        'sample_id': sample_ids,
        'G': sample_result.G,
        'A': sample_result.A,
        'total_attribution_mass': sample_result.w.sum(axis=1),
    })
