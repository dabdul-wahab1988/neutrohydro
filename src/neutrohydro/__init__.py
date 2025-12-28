"""
NeutroHydro: Neutrosophic Chemometrics for Groundwater Analysis

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
    # Data structures
    "NVIPResult",
    "STANDARD_MINERALS",
    # Version
    "__version__",
    "__author__",
    "__email__",
]
