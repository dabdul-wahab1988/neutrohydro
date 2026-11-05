"""
NeutroHydro: Neutralization-Displacement Geosystem (NDG) Framework

This package implements:
- NDG encoder: Maps scalar values to neutrosophic triplets (T, I, F)
- PNPLS: Probabilistic Neutrosophic PLS regression in augmented Hilbert space
- NVIP: Neutrosophic Variable Importance in Projection with L2 decomposition
- NSR/pi_G: Baseline vs perturbation attribution metrics
- Stoichiometric mineral inference via weighted NNLS
- Jacobian-orthogonal encoding with exact baseline/perturbation attribution
  (neutrohydro.orthogonal)
"""

from neutrohydro.preprocessing import Preprocessor
from neutrohydro.encoder import NDGEncoder
from neutrohydro.model import PNPLS
from neutrohydro.nvip import compute_nvip, NVIPResult
from neutrohydro.attribution import compute_nsr, compute_sample_baseline_fraction
from neutrohydro.minerals import MineralInverter, STANDARD_MINERALS
from neutrohydro.pipeline import NeutroHydroPipeline
from neutrohydro.orthogonal import (
    OrthogonalNDGEncoder,
    OrthogonalAttribution,
    orthogonal_attribution,
    subspace_alignment,
    jacobian_entanglement,
    verify_jacobian_orthogonality,
)
from neutrohydro.minerals import stoichiometric_identifiability

_VIZ_IMPORT_ERROR: Exception | None = None

try:
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
        plot_neutrosophic_ternary,
        plot_exact_r2_split,
        plot_ion_channel_shares,
        plot_process_alignment,
    )
except Exception as exc:  # optional deps: matplotlib/seaborn
    _VIZ_IMPORT_ERROR = exc

    def _viz_unavailable(*_args, **_kwargs):
        raise ImportError(
            "Visualization utilities require optional dependencies. "
            "Install with `pip install neutrohydro[viz]` (or `neutrohydro[all]`)."
        ) from _VIZ_IMPORT_ERROR

    plot_gibbs = _viz_unavailable
    plot_ilr_classification = _viz_unavailable
    plot_correlation_matrix = _viz_unavailable
    plot_mineral_fractions = _viz_unavailable
    plot_saturation_indices = _viz_unavailable
    plot_vip_decomposition = _viz_unavailable
    generate_report = _viz_unavailable
    mg_to_meq = _viz_unavailable
    classify_water_type = _viz_unavailable
    create_figure = _viz_unavailable
    PRESETS = {}
    plot_neutrosophic_ternary = _viz_unavailable
    plot_exact_r2_split = _viz_unavailable
    plot_ion_channel_shares = _viz_unavailable
    plot_process_alignment = _viz_unavailable

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
    "OrthogonalNDGEncoder",
    "OrthogonalAttribution",
    # Functions
    "compute_nvip",
    "compute_nsr",
    "compute_sample_baseline_fraction",
    "orthogonal_attribution",
    "subspace_alignment",
    "jacobian_entanglement",
    "verify_jacobian_orthogonality",
    "stoichiometric_identifiability",
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
    "plot_neutrosophic_ternary",
    "plot_exact_r2_split",
    "plot_ion_channel_shares",
    "plot_process_alignment",
    # Data structures
    "NVIPResult",
    "STANDARD_MINERALS",
    # Version
    "__version__",
    "__author__",
    "__email__",
]
