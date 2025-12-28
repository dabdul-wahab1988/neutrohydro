"""
Mineral stoichiometric inversion for NeutroHydro.

Implements Section 8 of the specification:
- Stoichiometric matrix for candidate minerals
- Weighted NNLS inversion using pi_G
- Mineral plausibility assessment
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls


# Standard ion order (meq/L recommended)
STANDARD_IONS = [
    "Ca2+",   # 0
    "Mg2+",   # 1
    "Na+",    # 2
    "K+",     # 3
    "HCO3-",  # 4
    "Cl-",    # 5
    "SO42-",  # 6
    "NO3-",   # 7
    "F-",     # 8
]

# Stoichiometric matrix: columns are minerals, rows are ions
# Values in meq/L per unit mineral dissolution
# Note: These are typical stoichiometries; adjust for specific conditions
STANDARD_MINERALS = {
    "Calcite": {
        "formula": "CaCO3",
        "stoichiometry": {
            "Ca2+": 2.0,    # 1 mol Ca2+ = 2 meq
            "HCO3-": 2.0,   # 1 mol HCO3- = 1 meq (approx)
        },
        "description": "Calcium carbonate dissolution",
    },
    "Dolomite": {
        "formula": "CaMg(CO3)2",
        "stoichiometry": {
            "Ca2+": 2.0,
            "Mg2+": 2.0,
            "HCO3-": 4.0,
        },
        "description": "Calcium-magnesium carbonate dissolution",
    },
    "Gypsum": {
        "formula": "CaSO4·2H2O",
        "stoichiometry": {
            "Ca2+": 2.0,
            "SO42-": 2.0,
        },
        "description": "Calcium sulfate dissolution",
    },
    "Halite": {
        "formula": "NaCl",
        "stoichiometry": {
            "Na+": 1.0,
            "Cl-": 1.0,
        },
        "description": "Sodium chloride dissolution",
    },
    "Sylvite": {
        "formula": "KCl",
        "stoichiometry": {
            "K+": 1.0,
            "Cl-": 1.0,
        },
        "description": "Potassium chloride dissolution",
    },
    "Anhydite": {
        "formula": "CaSO4",
        "stoichiometry": {
            "Ca2+": 2.0,
            "SO42-": 2.0,
        },
        "description": "Anhydrous calcium sulfate",
    },
    "Mirabilite": {
        "formula": "Na2SO4·10H2O",
        "stoichiometry": {
            "Na+": 2.0,
            "SO42-": 2.0,
        },
        "description": "Sodium sulfate dissolution",
    },
    "Fluorite": {
        "formula": "CaF2",
        "stoichiometry": {
            "Ca2+": 2.0,
            "F-": 2.0,
        },
        "description": "Calcium fluoride dissolution",
    },
    "Albite": {
        "formula": "NaAlSi3O8",
        "stoichiometry": {
            "Na+": 1.0,
            "HCO3-": 1.0,  # Simplified for weathering
        },
        "description": "Sodium feldspar weathering (simplified)",
    },
    "Kfeldspar": {
        "formula": "KAlSi3O8",
        "stoichiometry": {
            "K+": 1.0,
            "HCO3-": 1.0,  # Simplified for weathering
        },
        "description": "Potassium feldspar weathering (simplified)",
    },
}


def build_stoichiometric_matrix(
    minerals: dict,
    ion_order: list[str] = STANDARD_IONS,
) -> tuple[NDArray[np.floating], list[str], list[str]]:
    """
    Build stoichiometric matrix from mineral definitions.

    Parameters
    ----------
    minerals : dict
        Dictionary of mineral definitions with stoichiometry.
    ion_order : list of str
        Order of ions (rows of matrix).

    Returns
    -------
    A : ndarray of shape (n_ions, n_minerals)
        Stoichiometric matrix.
    mineral_names : list of str
        Names of minerals (column order).
    ion_names : list of str
        Names of ions (row order).
    """
    mineral_names = list(minerals.keys())
    m = len(ion_order)
    K = len(mineral_names)

    A = np.zeros((m, K))

    for k, mineral_name in enumerate(mineral_names):
        stoich = minerals[mineral_name]["stoichiometry"]
        for ion, value in stoich.items():
            if ion in ion_order:
                i = ion_order.index(ion)
                A[i, k] = value

    return A, mineral_names, ion_order


@dataclass
class MineralInversionResult:
    """Container for mineral inversion results."""

    s: NDArray[np.floating]          # Mineral contributions (n x K)
    residuals: NDArray[np.floating]  # Residual vectors (n x m)
    residual_norms: NDArray[np.floating]  # ||D(c - A*s)||_2 per sample
    plausible: NDArray[np.bool_]     # Mineral plausibility mask (n x K)
    mineral_fractions: NDArray[np.floating]  # Normalized fractions (n x K)


class MineralInverter:
    """
    Weighted NNLS-based mineral stoichiometric inverter.

    Uses baseline fraction (pi_G) to weight ions in the inversion,
    emphasizing baseline-dominant ions for mineral inference.

    Parameters
    ----------
    minerals : dict, optional
        Mineral definitions. If None, uses STANDARD_MINERALS.
    ion_order : list of str, optional
        Order of ions. If None, uses STANDARD_IONS.
    eta : float, default=1.0
        Weighting exponent: d_l = pi_G(ion_l)^eta
    tau_s : float, default=0.01
        Minimum contribution threshold for plausibility.
    tau_r : float, default=1.0
        Maximum weighted residual norm for plausibility.
    delta : float, default=1e-10
        Small constant for numerical stability.

    Attributes
    ----------
    A : ndarray of shape (n_ions, n_minerals)
        Stoichiometric matrix.
    mineral_names : list of str
        Names of candidate minerals.
    ion_names : list of str
        Names of ions in order.
    """

    def __init__(
        self,
        minerals: Optional[dict] = None,
        ion_order: Optional[list[str]] = None,
        eta: float = 1.0,
        tau_s: float = 0.01,
        tau_r: float = 1.0,
        delta: float = 1e-10,
    ):
        if minerals is None:
            minerals = STANDARD_MINERALS
        if ion_order is None:
            ion_order = STANDARD_IONS

        self.minerals = minerals
        self.ion_order = ion_order
        self.eta = eta
        self.tau_s = tau_s
        self.tau_r = tau_r
        self.delta = delta

        # Build stoichiometric matrix
        self.A, self.mineral_names, self.ion_names = build_stoichiometric_matrix(
            minerals, ion_order
        )
        self.n_ions = len(ion_order)
        self.n_minerals = len(self.mineral_names)

    def _compute_weights(
        self,
        pi_G: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute diagonal weights from pi_G.

        d_l = pi_G(ion_l)^eta

        Parameters
        ----------
        pi_G : ndarray of shape (n_ions,)
            Baseline fraction per ion.

        Returns
        -------
        D : ndarray of shape (n_ions,)
            Diagonal weights.
        """
        return np.power(pi_G + self.delta, self.eta)

    def invert(
        self,
        c: NDArray[np.floating],
        pi_G: Optional[NDArray[np.floating]] = None,
    ) -> MineralInversionResult:
        """
        Perform weighted NNLS inversion for mineral contributions.

        Solves: min_{s >= 0} ||D(c - As)||_2^2

        Parameters
        ----------
        c : ndarray of shape (n_samples, n_ions) or (n_ions,)
            Ion concentrations in meq/L.
        pi_G : ndarray of shape (n_ions,), optional
            Baseline fractions for weighting. If None, uses uniform weights.

        Returns
        -------
        result : MineralInversionResult
            Container with mineral contributions and diagnostics.
        """
        c = np.atleast_2d(c)
        n, m = c.shape

        if m != self.n_ions:
            raise ValueError(
                f"Expected {self.n_ions} ions, got {m}. "
                f"Ion order: {self.ion_names}"
            )

        # Compute weights
        if pi_G is None:
            D = np.ones(m)
        else:
            pi_G = np.asarray(pi_G)
            if len(pi_G) != m:
                raise ValueError(f"pi_G must have {m} elements")
            D = self._compute_weights(pi_G)

        # Weighted system: D @ A @ s ≈ D @ c
        # Transform: A_w = diag(D) @ A, c_w = D * c
        A_weighted = D[:, np.newaxis] * self.A

        # Storage
        s = np.zeros((n, self.n_minerals))
        residuals = np.zeros((n, m))
        residual_norms = np.zeros(n)

        for i in range(n):
            c_weighted = D * c[i, :]

            # Solve NNLS
            s_i, rnorm = nnls(A_weighted, c_weighted)
            s[i, :] = s_i

            # Compute residual in original space
            residual = c[i, :] - self.A @ s_i
            residuals[i, :] = residual

            # Weighted residual norm
            residual_norms[i] = np.linalg.norm(D * residual)

        # Plausibility assessment
        plausible = (s > self.tau_s) & (residual_norms[:, np.newaxis] <= self.tau_r)

        # Normalized mineral fractions
        s_total = s.sum(axis=1, keepdims=True) + self.delta
        mineral_fractions = s / s_total

        return MineralInversionResult(
            s=s,
            residuals=residuals,
            residual_norms=residual_norms,
            plausible=plausible,
            mineral_fractions=mineral_fractions,
        )

    def get_stoichiometry_dataframe(self):
        """
        Get stoichiometric matrix as pandas DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            Stoichiometric matrix with ion rows and mineral columns.
        """
        import pandas as pd

        return pd.DataFrame(
            self.A,
            index=self.ion_names,
            columns=self.mineral_names,
        )

    def results_to_dataframe(
        self,
        result: MineralInversionResult,
        sample_ids: Optional[list] = None,
    ):
        """
        Convert inversion results to pandas DataFrame.

        Parameters
        ----------
        result : MineralInversionResult
            Inversion results.
        sample_ids : list, optional
            Sample identifiers.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with mineral contributions per sample.
        """
        import pandas as pd

        n = result.s.shape[0]
        if sample_ids is None:
            sample_ids = list(range(n))

        data = {"sample_id": sample_ids, "residual_norm": result.residual_norms}

        for k, mineral in enumerate(self.mineral_names):
            data[f"{mineral}_s"] = result.s[:, k]
            data[f"{mineral}_frac"] = result.mineral_fractions[:, k]
            data[f"{mineral}_plausible"] = result.plausible[:, k]

        return pd.DataFrame(data)


def convert_to_meq(
    concentrations: NDArray[np.floating],
    ion_charges: dict[str, int],
    ion_masses: dict[str, float],
    from_unit: str = "mg/L",
) -> NDArray[np.floating]:
    """
    Convert ion concentrations to meq/L.

    Parameters
    ----------
    concentrations : ndarray of shape (n_samples, n_ions)
        Ion concentrations in original units.
    ion_charges : dict
        Absolute charge for each ion (e.g., {"Ca2+": 2, "Cl-": 1}).
    ion_masses : dict
        Molar mass in g/mol for each ion.
    from_unit : str
        Original unit: "mg/L" or "mmol/L".

    Returns
    -------
    concentrations_meq : ndarray
        Concentrations in meq/L.
    """
    concentrations = np.atleast_2d(concentrations)
    n, m = concentrations.shape

    result = np.zeros_like(concentrations)

    for j, (ion, charge) in enumerate(ion_charges.items()):
        mass = ion_masses[ion]
        if from_unit == "mg/L":
            # mg/L -> mmol/L -> meq/L
            # mmol/L = (mg/L) / (g/mol) = (mg/L) / mass
            # meq/L = mmol/L * |charge|
            result[:, j] = (concentrations[:, j] / mass) * abs(charge)
        elif from_unit == "mmol/L":
            # mmol/L -> meq/L
            result[:, j] = concentrations[:, j] * abs(charge)
        else:
            raise ValueError(f"Unknown unit: {from_unit}")

    return result


# Standard molar masses for common ions (g/mol)
ION_MASSES = {
    "Ca2+": 40.078,
    "Mg2+": 24.305,
    "Na+": 22.990,
    "K+": 39.098,
    "HCO3-": 61.017,
    "Cl-": 35.453,
    "SO42-": 96.06,
    "NO3-": 62.004,
    "F-": 18.998,
}

# Absolute charges for common ions
ION_CHARGES = {
    "Ca2+": 2,
    "Mg2+": 2,
    "Na+": 1,
    "K+": 1,
    "HCO3-": 1,
    "Cl-": 1,
    "SO42-": 2,
    "NO3-": 1,
    "F-": 1,
}
