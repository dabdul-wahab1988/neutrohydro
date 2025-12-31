"""
NeutroHydro: Neutralization-Displacement Geosystem (NDG) Framework

This package implements:
- NDG encoder: Maps scalar values to neutrosophic triplets (T, I, F)
- PNPLS: Probabilistic Neutrosophic PLS regression in augmented Hilbert space
- NVIP: Neutrosophic Variable Importance in Projection with L2 decomposition
- NSR/pi_G: Baseline vs perturbation attribution metrics
- Stoichiometric mineral inference via weighted NNLS
"""

from neutrohydro.preprocessing import Preprocessor
from neutrohydro.encoder import NDGEncoder
from neutrohydro.model import PNPLS
from neutrohydro.nvip import compute_nvip, NVIPResult
from neutrohydro.attribution import compute_nsr, compute_sample_baseline_fraction
from neutrohydro.minerals import MineralInverter, STANDARD_MINERALS
from neutrohydro.pipeline import NeutroHydroPipeline
from neutrohydro.visualization import (
    plot_gibbs,
    plot_ilr_classification,
    plot_correlation_matrix,
    plot_mineral_fractions,
    plot_saturation_indices,
    plot_vip_decomposition,
    generate_report,
    mg_to_meq,
    classify_water_type,
    create_figure,
    PRESETS,
)

__version__ = "1.0.0"
__author__ = "Dickson Abdul-Wahab"
__email__ = "dabdul-wahab@live.com"

__all__ = [
    # Core classes
    "Preprocessor",
    "NDGEncoder",
    "PNPLS",
    "NeutroHydroPipeline",
    "MineralInverter",
    # Functions
    "compute_nvip",
    "compute_nsr",
    "compute_sample_baseline_fraction",
    # Visualization
    "plot_gibbs",
    "plot_ilr_classification",
    "plot_correlation_matrix",
    "plot_mineral_fractions",
    "plot_saturation_indices",
    "plot_vip_decomposition",
    "generate_report",
    "mg_to_meq",
    "classify_water_type",
    "create_figure",
    "PRESETS",
    # Data structures
    "NVIPResult",
    "STANDARD_MINERALS",
    # Version
    "__version__",
    "__author__",
    "__email__",
]
