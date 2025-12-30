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


@dataclass
class MineralInversionResult:
    """Container for inversion results."""
    s: NDArray[np.floating]  # Raw mineral concentrations (meq/L)
    residuals: NDArray[np.floating]  # Residual ion concentrations
    residual_norms: NDArray[np.floating]  # Weighted residual norms
    plausible: NDArray[np.bool_]  # Boolean mask of plausible minerals
    mineral_fractions: NDArray[np.floating]  # Normalized fractions (0-1)
    indices: Optional[dict[str, NDArray[np.floating]]] = None # Hydrogeochemical Indices


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
    "Zn2+",   # 9  (Anthropogenic/Industrial)
    "Cd2+",   # 10 (Fertilizer impurity)
    "Pb2+",   # 11 (Industrial/Road)
    "B",      # 12 (Wastewater/Detergents)
    "Cu2+",   # 13 (Pesticides/Fungicides)
    "As",     # 14 (Pesticides/Geogenic)
    "Cr",     # 15 (Industrial)
    "U",      # 16 (Fertilizer impurity)
]

def calculate_cai(
    c: NDArray[np.floating],
    ion_names: list[str]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate Chloro-Alkaline Indices (CAI) to determine Ion Exchange direction.

    CAI-1 = [Cl - (Na + K)] / Cl
    CAI-2 = [Cl - (Na + K)] / (SO4 + HCO3 + CO3 + NO3)

    Interpretation:
    - Positive (> 0): Reverse Ion Exchange (Intrusion). Clay releases Ca/Mg, takes Na.
    - Negative (< 0): Normal Ion Exchange (Freshening). Clay releases Na, takes Ca/Mg.
    - Near Zero: No significant exchange (Simple mixing/dissolution).

    Parameters
    ----------
    c : ndarray (n_samples, n_ions)
        Concentrations in meq/L.
    ion_names : list[str]
        List of ion names matching columns of c.

    Returns
    -------
    cai1, cai2 : tuple of ndarrays
    """
    # Map indices
    try:
        idx_cl = ion_names.index("Cl-")
        idx_na = ion_names.index("Na+")
        idx_k = ion_names.index("K+")
        idx_so4 = ion_names.index("SO42-")
        idx_hco3 = ion_names.index("HCO3-")
        # Optional ions (handle if missing)
        idx_no3 = ion_names.index("NO3-") if "NO3-" in ion_names else None
    except ValueError as e:
        # If essential ions missing, return zeros
        return np.zeros(c.shape[0]), np.zeros(c.shape[0])

    cl = c[:, idx_cl]
    na = c[:, idx_na]
    k = c[:, idx_k]
    so4 = c[:, idx_so4]
    hco3 = c[:, idx_hco3]
    no3 = c[:, idx_no3] if idx_no3 is not None else 0.0

    # Numerator: Cl - (Na + K)
    numerator = cl - (na + k)

    # CAI-1 Denominator: Cl
    # Avoid divide by zero
    denom1 = cl.copy()
    denom1[denom1 == 0] = 1e-9
    cai1 = numerator / denom1

    # CAI-2 Denominator: Anions
    denom2 = so4 + hco3 + no3
    denom2[denom2 == 0] = 1e-9
    cai2 = numerator / denom2

    return cai1, cai2


def calculate_gibbs_ratios(
    c: NDArray[np.floating],
    ion_names: list[str]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate Gibbs Ratios for Hydrogeochemical Facies Classification.

    Ratio 1 (Anions): Cl / (Cl + HCO3)
    Ratio 2 (Cations): Na / (Na + Ca)

    Interpretation (Gibbs Diagram):
    - Low Ratio (< 0.5) + Moderate TDS: Rock Dominance (Weathering)
    - High Ratio (> 0.5) + High TDS: Evaporation Dominance (Salinization)
    - Low TDS: Precipitation Dominance (Rainfall)

    Parameters
    ----------
    c : ndarray
        Concentrations in meq/L.
    ion_names : list[str]
        List of ion names.

    Returns
    -------
    gibbs_anion, gibbs_cation : tuple of ndarrays
    """
    try:
        idx_cl = ion_names.index("Cl-")
        idx_hco3 = ion_names.index("HCO3-")
        idx_na = ion_names.index("Na+")
        idx_ca = ion_names.index("Ca2+")
    except ValueError:
        return np.zeros(c.shape[0]), np.zeros(c.shape[0])

    cl = c[:, idx_cl]
    hco3 = c[:, idx_hco3]
    na = c[:, idx_na]
    ca = c[:, idx_ca]

    # Avoid divide by zero
    denom_anion = cl + hco3
    denom_anion[denom_anion == 0] = 1e-9
    gibbs_anion = cl / denom_anion

    denom_cation = na + ca
    denom_cation[denom_cation == 0] = 1e-9
    gibbs_cation = na / denom_cation

    return gibbs_anion, gibbs_cation


def calculate_simpson_ratio(
    c: NDArray[np.floating],
    ion_names: list[str]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate Simpson's Ratio (Revelle Coefficient) and Freshening Ratio.

    SR = Cl / (HCO3 + CO3)
    FR = (HCO3 + CO3) / Cl  [Inverse form]

    Interpretation (SR):
    - < 0.5: Good quality
    - 0.5 - 1.3: Slightly contaminated
    - 1.3 - 2.8: Moderately contaminated
    - 2.8 - 6.6: Injuriously contaminated
    - 6.6 - 15.5: Highly contaminated

    Parameters
    ----------
    c : ndarray
        Concentrations in meq/L.
    ion_names : list[str]
        List of ion names.

    Returns
    -------
    simpson_ratio, freshening_ratio : tuple of ndarrays
    """
    try:
        idx_cl = ion_names.index("Cl-")
        idx_hco3 = ion_names.index("HCO3-")
        idx_co3 = ion_names.index("CO32-") if "CO32-" in ion_names else None
    except ValueError:
        n = c.shape[0]
        return np.zeros(n), np.zeros(n)

    cl = c[:, idx_cl]
    hco3 = c[:, idx_hco3]
    co3 = c[:, idx_co3] if idx_co3 is not None else 0.0

    # SR = Cl / (HCO3 + CO3)
    denom_sr = hco3 + co3
    denom_sr[denom_sr == 0] = 1e-9
    sr = cl / denom_sr

    # FR = (HCO3 + CO3) / Cl
    denom_fr = cl.copy()
    denom_fr[denom_fr == 0] = 1e-9
    fr = (hco3 + co3) / denom_fr

    return sr, fr


def calculate_bex(
    c: NDArray[np.floating],
    ion_names: list[str]
) -> NDArray[np.floating]:
    """
    Calculate Base Exchange Index (BEX) as a process-direction indicator.

    BEX = Na+ + K+ + Mg2+ - 1.0716 * Cl- (all in meq/L)

    Interpretation:
    - BEX > 0: Freshening trend
    - BEX < 0: Salinization trend
    - BEX ~ 0: No clear base-exchange signal

    Parameters
    ----------
    c : ndarray
    ion_names : list[str]

    Returns
    -------
    bex : ndarray
    """
    try:
        idx_na = ion_names.index("Na+")
        idx_k = ion_names.index("K+")
        idx_mg = ion_names.index("Mg2+")
        idx_cl = ion_names.index("Cl-")
    except ValueError:
        return np.zeros(c.shape[0])

    na = c[:, idx_na]
    k = c[:, idx_k]
    mg = c[:, idx_mg]
    cl = c[:, idx_cl]

    return na + k + mg - (1.0716 * cl)


def validate_chloride_origin(
    c: NDArray[np.floating],
    ion_names: list[str]
) -> NDArray[np.bool_]:
    """
    Check if Cl/Br ratio supports a Marine/Halite origin (Conservative Cl).

    Used to validate CAI constraints. If Cl/Br is low (< 200), it suggests
    anthropogenic inputs (sewage, etc.), making CAI unreliable.

    Parameters
    ----------
    c : ndarray
        Concentrations in meq/L.
    ion_names : list[str]
        List of ion names.

    Returns
    -------
    is_conservative : ndarray of bool
        True if Cl/Br ratio is consistent with Seawater/Halite or if Br is missing.
        False if Cl/Br ratio suggests Anthropogenic input (< 200).
    """
    # Find indices
    try:
        idx_cl = ion_names.index("Cl-")
        # Check for Br (could be Br or Br-)
        if "Br" in ion_names:
            idx_br = ion_names.index("Br")
        elif "Br-" in ion_names:
            idx_br = ion_names.index("Br-")
        else:
            return np.ones(c.shape[0], dtype=bool)  # No Br, assume valid
    except ValueError:
        return np.ones(c.shape[0], dtype=bool)

    cl_meq = c[:, idx_cl]
    br_meq = c[:, idx_br]

    # Convert to Mass (mg/L)
    # Cl: 35.45 mg/mmol. Valence 1.
    # Br: 79.90 mg/mmol. Valence 1.
    cl_mg = cl_meq * 35.45
    br_mg = br_meq * 79.90

    # Avoid divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = cl_mg / br_mg

    # Threshold: Anthropogenic usually < 200. Seawater ~290.
    # We use 200 as a conservative cutoff.
    is_valid = np.ones(c.shape[0], dtype=bool)

    # Only invalidate if we have data (Br > 0) and ratio is low
    has_data = (br_mg > 0) & (cl_mg > 0)
    # If ratio is NaN/Inf, we assume valid (no evidence of pollution)
    # We only flag if we have positive proof of low ratio
    if np.any(has_data):
        is_valid[has_data] = ratio[has_data] >= 200.0

    return is_valid


# --- ION EXCHANGE PHASES (Optional) ---
# These represent non-stoichiometric exchange on clay surfaces.
# They are NOT included in STANDARD_MINERALS by default to preserve
# strict mass balance, but can be added by the user for advanced modeling.
#
# Concept:
# Na-Ca Exchange (Freshening): Ca2+ (water) + 2Na-X (clay) -> Ca-X2 (clay) + 2Na+ (water)
#   - Water gains 2 Na+, loses 1 Ca2+
#   - Stoichiometry: Na+ = +1.0, Ca2+ = -1.0 (normalized to meq)
#
# Ca-Na Exchange (Intrusion): 2Na+ (water) + Ca-X2 (clay) -> 2Na-X (clay) + Ca2+ (water)
#   - Water gains 1 Ca2+, loses 2 Na+
#   - Stoichiometry: Ca2+ = +1.0, Na+ = -1.0 (normalized to meq)

EXCHANGER_PHASES = {
    "Clay_ReleaseNa": {
        "formula": "Na-Ca-Exchanger",
        "stoichiometry": {
            "Na+": 1.0,
            "Ca2+": -1.0,
        },
        "description": "Ion Exchange: Release Na, Adsorb Ca (Freshening)",
    },
    "Clay_ReleaseCa": {
        "formula": "Ca-Na-Exchanger",
        "stoichiometry": {
            "Ca2+": 1.0,
            "Na+": -1.0,
        },
        "description": "Ion Exchange: Release Ca, Adsorb Na (Intrusion)",
    },
    "Clay_ReleaseMg": {
        "formula": "Mg-Ca-Exchanger",
        "stoichiometry": {
            "Mg2+": 1.0,
            "Ca2+": -1.0,
        },
        "description": "Ion Exchange: Release Mg, Adsorb Ca",
    }
}

# --- REDOX PHASES (Optional) ---
# These represent biogeochemical sinks (mass loss) or sources (mass gain)
# driven by redox reactions.
#
# Concept:
# Denitrification: NO3- -> N2(g) (Loss of Nitrate)
#   - Water loses NO3-
#   - Stoichiometry: NO3- = -1.0 (Sink)
#
# Sulfate Reduction: SO4-- -> H2S(g) (Loss of Sulfate)
#   - Water loses SO4--
#   - Stoichiometry: SO42- = -1.0 (Sink)

REDOX_PHASES = {
    "Sink_Denitrification": {
        "formula": "NO3-Reduction",
        "stoichiometry": {
            "NO3-": -1.0,
            "HCO3-": 1.0, # Often produces alkalinity
        },
        "description": "Redox: Denitrification (Nitrate Loss)",
    },
    "Sink_SulfateReduction": {
        "formula": "SO4-Reduction",
        "stoichiometry": {
            "SO42-": -1.0,
            "HCO3-": 1.0, # Often produces alkalinity
        },
        "description": "Redox: Sulfate Reduction (Sulfate Loss)",
    },
    "Source_Nitrification": {
        "formula": "NH4-Oxidation",
        "stoichiometry": {
            "NO3-": 1.0,
            # Consumes alkalinity, but maybe keep simple for now
        },
        "description": "Redox: Nitrification (Nitrate Gain)",
    }
}

# Stoichiometric matrix: columns are minerals, rows are ions
# Values in meq/L per unit mineral dissolution
# Note: These are typical stoichiometries; adjust for specific conditions
STANDARD_MINERALS = {
    "Calcite": {
        "formula": "CaCO3",
        "type": "weathering",
        "stoichiometry": {
            "Ca2+": 2.0,    # 1 mol Ca2+ = 2 meq
            "HCO3-": 2.0,   # 1 mol HCO3- = 1 meq (approx)
        },
        "description": "Calcium carbonate dissolution",
    },
    "Dolomite": {
        "formula": "CaMg(CO3)2",
        "type": "weathering",
        "stoichiometry": {
            "Ca2+": 2.0,
            "Mg2+": 2.0,
            "HCO3-": 4.0,
        },
        "description": "Calcium-magnesium carbonate dissolution",
    },
    "Magnesite": {
        "formula": "MgCO3",
        "type": "weathering",
        "stoichiometry": {
            "Mg2+": 2.0,
            "HCO3-": 2.0,
        },
        "description": "Magnesium carbonate dissolution",
    },
    "Gypsum": {
        "formula": "CaSO4·2H2O",
        "type": "evaporite",
        "stoichiometry": {
            "Ca2+": 2.0,
            "SO42-": 2.0,
        },
        "description": "Calcium sulfate dissolution",
    },
    "Anhydrite": {
        "formula": "CaSO4",
        "type": "evaporite",
        "stoichiometry": {
            "Ca2+": 2.0,
            "SO42-": 2.0,
        },
        "description": "Anhydrous calcium sulfate",
    },
    "Halite": {
        "formula": "NaCl",
        "type": "evaporite",
        "stoichiometry": {
            "Na+": 1.0,
            "Cl-": 1.0,
        },
        "description": "Sodium chloride dissolution",
    },
    "Sylvite": {
        "formula": "KCl",
        "type": "evaporite",
        "stoichiometry": {
            "K+": 1.0,
            "Cl-": 1.0,
        },
        "description": "Potassium chloride dissolution",
    },
    "Mirabilite": {
        "formula": "Na2SO4·10H2O",
        "type": "evaporite",
        "stoichiometry": {
            "Na+": 2.0,
            "SO42-": 2.0,
        },
        "description": "Sodium sulfate dissolution",
    },
    "Thenardite": {
        "formula": "Na2SO4",
        "type": "evaporite",
        "stoichiometry": {
            "Na+": 2.0,
            "SO42-": 2.0,
        },
        "description": "Anhydrous sodium sulfate",
    },
    "Glauberite": {
        "formula": "Na2Ca(SO4)2",
        "type": "evaporite",
        "stoichiometry": {
            "Na+": 2.0,
            "Ca2+": 2.0,
            "SO42-": 4.0,
        },
        "description": "Sodium-calcium sulfate dissolution",
    },
    "Epsomite": {
        "formula": "MgSO4·7H2O",
        "type": "evaporite",
        "stoichiometry": {
            "Mg2+": 2.0,
            "SO42-": 2.0,
        },
        "description": "Magnesium sulfate dissolution",
    },
    "Fluorite": {
        "formula": "CaF2",
        "type": "weathering",
        "stoichiometry": {
            "Ca2+": 2.0,
            "F-": 2.0,
        },
        "description": "Calcium fluoride dissolution",
    },
    "Albite": {
        "formula": "NaAlSi3O8",
        "type": "weathering",
        "stoichiometry": {
            "Na+": 1.0,
            "HCO3-": 1.0,  # Incongruent dissolution to kaolinite
        },
        "description": "Sodium feldspar weathering",
    },
    "Anorthite": {
        "formula": "CaAl2Si2O8",
        "type": "weathering",
        "stoichiometry": {
            "Ca2+": 2.0,
            "HCO3-": 2.0,  # Incongruent dissolution to kaolinite
        },
        "description": "Calcium feldspar weathering",
    },
    "Kfeldspar": {
        "formula": "KAlSi3O8",
        "type": "weathering",
        "stoichiometry": {
            "K+": 1.0,
            "HCO3-": 1.0,  # Incongruent dissolution to kaolinite
        },
        "description": "Potassium feldspar weathering",
    },
    "Biotite": {
        "formula": "KMg3AlSi3O10(OH)2",
        "stoichiometry": {
            "K+": 1.0,
            "Mg2+": 6.0,
            "HCO3-": 7.0,
        },
        "description": "Biotite weathering (Mg-endmember)",
    },
    "Niter": {
        "formula": "KNO3",
        "stoichiometry": {
            "K+": 1.0,
            "NO3-": 1.0,
        },
        "description": "Potassium nitrate dissolution",
    },
    "SodaNiter": {
        "formula": "NaNO3",
        "stoichiometry": {
            "Na+": 1.0,
            "NO3-": 1.0,
        },
        "description": "Sodium nitrate dissolution",
    },
    "Nitrocalcite": {
        "formula": "Ca(NO3)2·4H2O",
        "stoichiometry": {
            "Ca2+": 2.0,
            "NO3-": 2.0,
        },
        "description": "Calcium nitrate dissolution (Fertilizer)",
    },
    # --- Anthropogenic / Trace Markers ---
    "Otavite": {
        "formula": "CdCO3",
        "stoichiometry": {
            "Cd2+": 2.0,
            "HCO3-": 2.0,
        },
        "description": "Cadmium carbonate (Fertilizer impurity marker)",
    },
    "Smithsonite": {
        "formula": "ZnCO3",
        "stoichiometry": {
            "Zn2+": 2.0,
            "HCO3-": 2.0,
        },
        "description": "Zinc carbonate (Industrial/Sewage marker)",
    },
    "Cerussite": {
        "formula": "PbCO3",
        "stoichiometry": {
            "Pb2+": 2.0,
            "HCO3-": 2.0,
        },
        "description": "Lead carbonate (Industrial/Road marker)",
    },
    "Borax": {
        "formula": "Na2B4O7·10H2O",
        "stoichiometry": {
            "Na+": 2.0,
            "B": 4.0,  # Molar ratio
        },
        "description": "Borax (Wastewater/Detergent marker)",
    },
    "Malachite": {
        "formula": "Cu2CO3(OH)2",
        "stoichiometry": {
            "Cu2+": 4.0,
            "HCO3-": 4.0,
        },
        "description": "Copper carbonate (Pesticide/Industrial marker)",
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
        use_cai_constraints: bool = True,
        use_gibbs_constraints: bool = True,
        quality_flags: Optional[list[dict]] = None,
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
        use_cai_constraints : bool, default=True
            If True, calculates Chloro-Alkaline Indices (CAI) for each sample
            and dynamically restricts Ion Exchange phases.
        use_gibbs_constraints : bool, default=True
            If True, calculates Gibbs Ratios to classify samples as Rock Dominance
            or Evaporation Dominance.
        quality_flags : list[dict], optional
            List of quality assessment dictionaries (from quality_check.py).
            Used to override constraints or force plausibility based on pollution sources.

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

        # Calculate CAI constraints if requested
        cai_mask_release_na = np.ones(n, dtype=bool) # Allowed by default
        cai_mask_release_ca = np.ones(n, dtype=bool) # Allowed by default

        if use_cai_constraints:
            cai1, _ = calculate_cai(c, self.ion_names)
            
            # Validate Chloride Origin using Cl/Br ratio (if available)
            # If Cl/Br < 200 (Anthropogenic), CAI is unreliable -> Trust = False
            cai_reliable = validate_chloride_origin(c, self.ion_names)
            
            # CAI < 0: Normal Exchange (Release Na). Ban Release Ca.
            # CAI > 0: Reverse Exchange (Release Ca). Ban Release Na.
            # Use a small threshold for numerical noise around 0
            threshold = 0.05
            
            # Only enforce constraints if CAI is reliable
            # If unreliable (Anthropogenic Cl), we allow ALL exchange directions
            # because we can't trust the index.
            
            # Case 1: Freshening Signal (CAI < -threshold) AND Reliable
            # Implication: ReleaseNa is happening. ReleaseCa is impossible.
            mask_freshening = (cai1 < -threshold) & cai_reliable
            cai_mask_release_ca[mask_freshening] = False  # Ban ReleaseCa
            
            # Case 2: Intrusion Signal (CAI > threshold) AND Reliable
            # Implication: ReleaseCa is happening. ReleaseNa is impossible.
            mask_intrusion = (cai1 > threshold) & cai_reliable
            cai_mask_release_na[mask_intrusion] = False   # Ban ReleaseNa

        # Calculate Gibbs Constraints if requested
        gibbs_mask_rock = np.zeros(n, dtype=bool)
        if use_gibbs_constraints:
            g_anion, g_cation = calculate_gibbs_ratios(c, self.ion_names)
            # Simple classification: If both ratios < 0.5, it's Rock Dominance
            # (or Rainfall, but Rainfall also has low ratios).
            # In Rock Dominance, Evaporites are unlikely.
            gibbs_mask_rock = (g_anion < 0.5) & (g_cation < 0.5)

        # Identify indices of exchanger phases in the matrix A
        # This assumes the user added them to the inverter.
        idx_release_na = -1
        idx_release_ca = -1
        for k, name in enumerate(self.mineral_names):
            if "ReleaseNa" in name: idx_release_na = k
            if "ReleaseCa" in name: idx_release_ca = k
            
        # Identify indices of Evaporite minerals
        # We check the 'type' field in STANDARD_MINERALS if available
        # Or infer from name list if passed directly
        idx_evaporites = []
        for k, name in enumerate(self.mineral_names):
            # Check if name exists in STANDARD_MINERALS and has type 'evaporite'
            if name in STANDARD_MINERALS and STANDARD_MINERALS[name].get("type") == "evaporite":
                idx_evaporites.append(k)

        # Weighted system: D @ A @ s ≈ D @ c
        # Transform: A_w = diag(D) @ A, c_w = D * c
        A_weighted_base = D[:, np.newaxis] * self.A

        # Storage
        s = np.zeros((n, self.n_minerals))
        residuals = np.zeros((n, m))
        residual_norms = np.zeros(n)

        for i in range(n):
            c_weighted = D * c[i, :]
            
            # Dynamic Matrix Construction per Sample
            # Start with the full weighted matrix
            A_i = A_weighted_base.copy()
            
            # Apply Gibbs Constraints (Soft Penalty)
            # If Rock Dominance, scale down the columns of Evaporite minerals
            # This effectively penalizes them in the NNLS solve (requires more mass to fit same signal)
            # Actually, scaling column A_j by factor f means s_j must be s_j/f to have same effect.
            # Wait, NNLS minimizes ||Ax - b||.
            # If we scale column A_j by 10, then s_j will be 1/10th to fit the same data.
            # We want to DISCOURAGE usage.
            # To discourage, we can't easily use column scaling in NNLS without regularization.
            # Alternative: Hard Masking (Remove them).
            # Let's use Hard Masking for strictness, as requested ("help the model").
            # If Rock Dominance, we assume NO Halite/Gypsum unless explicitly forced.
            # But wait, Rock waters have SOME Cl.
            # Let's use a "Threshold Masking": If Gibbs says Rock, we remove Evaporites.
            # The Cl will be left as Residual (or fitted by Sylvite/Halite if we leave one).
            # Better: Leave Halite (ubiquitous) but remove others (Mirabilite, Thenardite, Gypsum).
            # Let's stick to the user's request: "help the model".
            # Helping means reducing ambiguity.
            # Let's remove "Complex Evaporites" (Sulfates) in Rock Dominance.
            
            active_indices = list(range(self.n_minerals))
            
            # Apply CAI Constraints
            if use_cai_constraints:
                if idx_release_na != -1 and not cai_mask_release_na[i]:
                    if idx_release_na in active_indices: active_indices.remove(idx_release_na)
                if idx_release_ca != -1 and not cai_mask_release_ca[i]:
                    if idx_release_ca in active_indices: active_indices.remove(idx_release_ca)
            
            # Apply Gibbs Constraints
            if use_gibbs_constraints and gibbs_mask_rock[i]:
                # In Rock Dominance, remove rare evaporites.
                # Keep Halite (NaCl) as it's common everywhere.
                # Remove Sulfates (Gypsum, Mirabilite, etc.) unless they are the ONLY source of SO4.
                # This is tricky. Let's just remove the "Sodium Sulfates" (Mirabilite, Thenardite)
                # as they are strictly evaporitic. Gypsum can be weathering.
                for k_evap in idx_evaporites:
                    name = self.mineral_names[k_evap]
                    # Filter out Sodium Sulfates in Rock waters
                    if name in ["Mirabilite", "Thenardite", "Glauberite", "Epsomite"]:
                        if k_evap in active_indices: active_indices.remove(k_evap)

            # Apply Quality Flag Overrides
            if quality_flags and i < len(quality_flags):
                q_data = quality_flags[i]
                sources = q_data.get("Inferred_Sources", "")
                
                # If specific pollution sources are identified, we ensure relevant minerals are ACTIVE
                # This overrides previous exclusions (e.g. from Gibbs)
                
                # 1. Saline Intrusion -> Ensure Halite/Sylvite are active
                if "Saline Intrusion" in sources:
                    for m_name in ["Halite", "Sylvite"]:
                        if m_name in self.mineral_names:
                            idx = self.mineral_names.index(m_name)
                            if idx not in active_indices: active_indices.append(idx)
                            
                # 2. Gypsum/Evaporites -> Ensure Sulfates are active
                if "Gypsum" in sources:
                    for m_name in ["Gypsum", "Anhydrite", "Mirabilite", "Thenardite"]:
                        if m_name in self.mineral_names:
                            idx = self.mineral_names.index(m_name)
                            if idx not in active_indices: active_indices.append(idx)

            # Slice A for active minerals
            A_active = A_i[:, active_indices]
            
            # Solve NNLS
            s_active, rnorm = nnls(A_active, c_weighted)
            
            # Map back to full s vector
            s_i = np.zeros(self.n_minerals)
            s_i[active_indices] = s_active
            
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

        # Calculate Indices for reporting
        cai1, cai2 = calculate_cai(c, self.ion_names)
        gibbs_anion, gibbs_cation = calculate_gibbs_ratios(c, self.ion_names)
        sr, fr = calculate_simpson_ratio(c, self.ion_names)
        bex = calculate_bex(c, self.ion_names)
        
        # Add Simpson Class
        simpson_class = []
        for val in sr:
            # Thresholds based on documentation (Todd/Simpson classes)
            if val < 0.5: simpson_class.append("Good quality")
            elif val < 1.3: simpson_class.append("Slightly contaminated")
            elif val < 2.8: simpson_class.append("Moderately contaminated")
            elif val < 6.6: simpson_class.append("Injuriously contaminated")
            elif val < 15.5: simpson_class.append("Highly contaminated")
            else: simpson_class.append("Extremely contaminated")

        return MineralInversionResult(
            s=s,
            residuals=residuals,
            residual_norms=residual_norms,
            plausible=plausible,
            mineral_fractions=mineral_fractions,
            indices={
                "CAI_1": cai1,
                "CAI_2": cai2,
                "Gibbs_Anion": gibbs_anion,
                "Gibbs_Cation": gibbs_cation,
                "Simpson_Ratio": sr,
                "Freshening_Ratio": fr,
                "Simpson_Class": simpson_class,
                "BEX": bex
            }
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
            
        # Add Indices if available
        if result.indices:
            for idx_name, idx_vals in result.indices.items():
                data[idx_name] = idx_vals

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

    # If ion_charges has more items than columns, we assume the user passed
    # a subset of ions corresponding to the first m keys, OR we should rely
    # on the user passing the correct ordered dict.
    #
    # To be safe and robust: We iterate up to min(m, len(ion_charges)).
    
    keys = list(ion_charges.keys())
    if len(keys) != m:
        # If dimensions mismatch, we can't guess the mapping without names.
        # But assuming the caller prepared the matrix in the same order as the dict keys
        # is the standard contract here.
        # We will iterate over the columns we have.
        pass

    for j in range(m):
        if j >= len(keys):
            break # No more charge info available
            
        ion = keys[j]
        charge = ion_charges[ion]
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
    "Zn2+": 65.38,
    "Cd2+": 112.41,
    "Pb2+": 207.2,
    "B": 10.81,
    "Cu2+": 63.546,
    "As": 74.92,
    "Cr": 51.996,
    "U": 238.03,
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
    "Zn2+": 2,
    "Cd2+": 2,
    "Pb2+": 2,
    "B": 1,      # Treated as molar equivalent
    "Cu2+": 2,
    "As": 3,     # Assumed As(III)
    "Cr": 3,     # Assumed Cr(III)
    "U": 6,      # Assumed U(VI)
}
