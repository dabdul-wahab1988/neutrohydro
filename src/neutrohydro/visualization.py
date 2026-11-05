"""
Visualization module for NeutroHydro.

Provides publication-quality hydrogeochemical plots including:
- Gibbs Diagrams (cation/anion process identification)
- ILR Water Classification (compositional 2x2 plot)
- Mineral Fraction Charts
- Saturation Index Plots
- VIP Decomposition Plots
- Correlation Matrices

Uses a modular plot registry for flexible figure composition.
"""

import os
import re
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import seaborn as sns


def _missing_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def _require_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
    missing = _missing_columns(df, required)
    if missing:
        raise ValueError(
            f"{context} requires columns: {required}. Missing: {missing}."
        )


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=float)
    np.divide(num, den, out=out, where=(den != 0) & np.isfinite(den) & np.isfinite(num))
    return out


def _safe_log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    np.log(x, out=out, where=(x > 0) & np.isfinite(x))
    return out

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

PLOT_STYLE = {
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
}

def apply_publication_style():
    """Apply publication-quality matplotlib settings."""
    plt.rcParams.update(PLOT_STYLE)
    sns.set_style("whitegrid")

# =============================================================================
# ION/MINERAL MAPPINGS
# =============================================================================

ION_MAP = {
    'Ca': r'Ca$^{2+}$', 'Mg': r'Mg$^{2+}$', 'Na': r'Na$^{+}$', 'K': r'K$^{+}$',
    'HCO3': r'HCO$_3^{-}$', 'Cl': r'Cl$^{-}$', 'SO4': r'SO$_4^{2-}$', 'NO3': r'NO$_3^{-}$', 'F': r'F$^{-}$'
}

# Molar masses for unit conversion (g/mol)
MOLAR_MASS = {
    'Ca': 40.08, 'Mg': 24.31, 'Na': 22.99, 'K': 39.10,
    'HCO3': 61.02, 'Cl': 35.45, 'SO4': 96.06, 'NO3': 62.00, 'F': 19.00
}

# Equivalent factors (charge / molar mass)
EQUIV_FACTOR = {
    'Ca': 2 / 40.08, 'Mg': 2 / 24.31, 'Na': 1 / 22.99, 'K': 1 / 39.10,
    'HCO3': 1 / 61.02, 'Cl': 1 / 35.45, 'SO4': 2 / 96.06, 'NO3': 1 / 62.00, 'F': 1 / 19.00
}

# =============================================================================
# PLOT REGISTRY
# =============================================================================

PLOT_REGISTRY = {
    # Gibbs Diagrams
    'g1': {'func': '_plot_gibbs_cations', 'type': 'gibbs', 'name': 'Gibbs Cations'},
    'g2': {'func': '_plot_gibbs_anions', 'type': 'gibbs', 'name': 'Gibbs Anions'},
    
    # ILR Classification (single 2x2 figure)
    'i1': {'func': '_plot_ilr_classification', 'type': 'ilr', 'name': 'ILR Water Classification'},
    
    # Hydrochemistry Diagnostics
    'h1': {'func': '_plot_correlation_matrix', 'type': 'diagnostic', 'name': 'Correlation Matrix'},
    
    # Mineral Inversion Results
    'm1': {'func': '_plot_mineral_fractions', 'type': 'mineral', 'name': 'Mineral Fractions'},
    'm2': {'func': '_plot_saturation_indices', 'type': 'mineral', 'name': 'Saturation Indices'},
    'm3': {'func': '_plot_mineral_plausibility', 'type': 'mineral', 'name': 'Thermo Plausibility'},
    
    # VIP / Attribution
    'v1': {'func': '_plot_vip_decomposition', 'type': 'vip', 'name': 'VIP Decomposition'},
    'v2': {'func': '_plot_vip_aggregate', 'type': 'vip', 'name': 'VIP Aggregate'},
    'v3': {'func': '_plot_baseline_fraction', 'type': 'vip', 'name': 'Baseline Fraction (πG)'},
    'v4': {'func': '_plot_g_histogram', 'type': 'vip', 'name': 'Sample G Distribution'},

    # Model Diagnostics
    'p1': {'func': '_plot_pls_loadings', 'type': 'model', 'name': 'PLS Component Loadings'},
    'p2': {'func': '_plot_explained_variance', 'type': 'model', 'name': 'Explained Variance'},
}

PRESETS = {
    'fig1_classification': '[g1|g2][i1]',
    'fig2_correlation': '[h1]',
    'fig3_minerals': '[m1|m2][m3]',
    'fig4_attribution': '[v1|v2][v3|v4]',
    'fig5_model': '[p1][p2]',
}

# =============================================================================
# UNIT CONVERSION
# =============================================================================

def mg_to_meq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert major ions from mg/L to meq/L.
    
    Parameters
    ----------
    df : DataFrame
        Input data with ion columns in mg/L (Ca, Mg, Na, K, HCO3, Cl, SO4, NO3, F)
    
    Returns
    -------
    df_meq : DataFrame
        Copy of input with additional columns: Ca_meq, Mg_meq, etc.
    """
    df_meq = df.copy()
    
    for ion, factor in EQUIV_FACTOR.items():
        if ion in df.columns:
            df_meq[f'{ion}_meq'] = df[ion] * factor
    
    return df_meq


def calculate_ion_sums(df_meq: pd.DataFrame) -> pd.DataFrame:
    """Calculate cation/anion sums and ion balance error."""
    cation_cols = [c for c in ['Ca_meq', 'Mg_meq', 'Na_meq', 'K_meq'] if c in df_meq.columns]
    anion_cols = [c for c in ['HCO3_meq', 'Cl_meq', 'SO4_meq', 'NO3_meq', 'F_meq'] if c in df_meq.columns]
    
    df_meq['Sum_Cations'] = df_meq[cation_cols].sum(axis=1)
    df_meq['Sum_Anions'] = df_meq[anion_cols].sum(axis=1)
    df_meq['Ion_Balance_Error'] = 100 * (df_meq['Sum_Cations'] - df_meq['Sum_Anions']) / (
        df_meq['Sum_Cations'] + df_meq['Sum_Anions']
    )
    
    return df_meq


# =============================================================================
# WATER TYPE CLASSIFICATION
# =============================================================================

def classify_water_type(
    row: pd.Series,
    cation_threshold: float = 0.5,
    anion_threshold: float = 0.5
) -> str:
    """
    Classify water type based on ion dominance.
    
    Parameters
    ----------
    row : Series
        Row with _meq columns for major ions
    cation_threshold : float
        Threshold for dominant cation fraction
    anion_threshold : float
        Threshold for dominant anion fraction
    
    Returns
    -------
    water_type : str
        Classification like 'Ca-HCO₃', 'Na-Cl', etc.
    """
    # Cations: Ca, Mg, Na, K
    ca = float(row.get('Ca_meq', 0.0) or 0.0)
    mg = float(row.get('Mg_meq', 0.0) or 0.0)
    na = float(row.get('Na_meq', 0.0) or 0.0)
    k = float(row.get('K_meq', 0.0) or 0.0)
    total_cations = ca + mg + na + k
    if not np.isfinite(total_cations) or total_cations <= 0:
        return 'Mixed'

    ca_frac = ca / total_cations
    mg_frac = mg / total_cations
    na_frac = na / total_cations
    k_frac = k / total_cations

    cations = []
    if ca_frac > cation_threshold or ca_frac > 0.25:
        cations.append('Ca')
    if mg_frac > cation_threshold or mg_frac > 0.25:
        cations.append('Mg')
    if na_frac > cation_threshold or na_frac > 0.25:
        cations.append('Na')
    if k_frac > cation_threshold or k_frac > 0.25:
        cations.append('K')

    # Anions: HCO3, Cl, SO4 (major anions only)
    hco3 = float(row.get('HCO3_meq', 0.0) or 0.0)
    cl = float(row.get('Cl_meq', 0.0) or 0.0)
    so4 = float(row.get('SO4_meq', 0.0) or 0.0)
    total_anions = hco3 + cl + so4
    if not np.isfinite(total_anions) or total_anions <= 0:
        return 'Mixed'

    hco3_frac = hco3 / total_anions
    cl_frac = cl / total_anions
    so4_frac = so4 / total_anions

    anions = []
    if hco3_frac > anion_threshold or hco3_frac > 0.25:
        anions.append('HCO$_3$')
    if cl_frac > anion_threshold or cl_frac > 0.25:
        anions.append('Cl')
    if so4_frac > anion_threshold or so4_frac > 0.25:
        anions.append('SO$_4$')

    water_type = '-'.join(cations + anions) if cations and anions else 'Mixed'
    return water_type


# =============================================================================
# GIBBS PLOTS
# =============================================================================

def plot_gibbs(
    df: pd.DataFrame,
    df_meq: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Create dual-panel Gibbs plots (Cations and Anions).
    
    Parameters
    ----------
    df : DataFrame
        Raw ion data with TDS column
    df_meq : DataFrame, optional
        Precomputed meq/L data. If None, computed from df.
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()
    
    if df_meq is None:
        df_meq = mg_to_meq(df)

    _require_columns(df, ['TDS'], 'plot_gibbs')
    _require_columns(df_meq, ['Na_meq', 'Ca_meq', 'Cl_meq', 'HCO3_meq'], 'plot_gibbs')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    _plot_gibbs_cations(ax1, df, df_meq)
    _plot_gibbs_anions(ax2, df, df_meq)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def _plot_gibbs_cations(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    df_meq: pd.DataFrame,
    **kwargs
) -> None:
    """Render Gibbs cation plot onto provided axes."""
    tds = df['TDS']
    na = df_meq['Na_meq'].to_numpy(dtype=float)
    ca = df_meq['Ca_meq'].to_numpy(dtype=float)

    na_ratio = _safe_divide(na, na + ca)
    
    ax.scatter(na_ratio, tds, c='steelblue', edgecolor='black', s=80, alpha=0.7)
    ax.set_yscale('log')
    ax.set_ylim(1, 100000)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel(r'Na$^{+}$ / (Na$^{+}$ + Ca$^{2+}$)', fontsize=12)
    ax.set_ylabel('TDS (mg/L)', fontsize=12)
    ax.set_title('Gibbs Plot - Cations', fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Process domain labels
    ax.text(0.9, 20, 'Precipitation\nDominance', fontsize=10, ha='center', va='center')
    ax.text(0.25, 500, 'Rock Weathering\nDominance', fontsize=10, ha='center', va='center')
    ax.text(0.9, 5000, 'Evaporation\nDominance', fontsize=10, ha='center', va='top')
    
    # Panel label
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')


def _plot_gibbs_anions(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    df_meq: pd.DataFrame,
    **kwargs
) -> None:
    """Render Gibbs anion plot onto provided axes."""
    tds = df['TDS']
    cl = df_meq['Cl_meq'].to_numpy(dtype=float)
    hco3 = df_meq['HCO3_meq'].to_numpy(dtype=float)

    cl_ratio = _safe_divide(cl, cl + hco3)
    
    ax.scatter(cl_ratio, tds, c='steelblue', edgecolor='black', s=80, alpha=0.7)
    ax.set_yscale('log')
    ax.set_ylim(1, 100000)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel(r'Cl$^{-}$ / (Cl$^{-}$ + HCO$_3^{-}$)', fontsize=12)
    ax.set_ylabel('TDS (mg/L)', fontsize=12)
    ax.set_title('Gibbs Plot - Anions', fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Process domain labels
    ax.text(0.9, 20, 'Precipitation\nDominance', fontsize=10, ha='center', va='center')
    ax.text(0.25, 500, 'Rock Weathering\nDominance', fontsize=10, ha='center', va='center')
    ax.text(0.9, 5000, 'Evaporation\nDominance', fontsize=10, ha='center', va='top')
    
    # Panel label
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')


# =============================================================================
# ILR WATER CLASSIFICATION
# =============================================================================

def plot_ilr_classification(
    df: pd.DataFrame,
    df_meq: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Create 2x2 ILR (Isometric Log-Ratio) water classification plot.
    
    Parameters
    ----------
    df : DataFrame
        Raw ion data
    df_meq : DataFrame, optional
        Precomputed meq/L data
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()
    
    if df_meq is None:
        df_meq = mg_to_meq(df)

    _require_columns(
        df_meq,
        ['Ca_meq', 'Mg_meq', 'Na_meq', 'K_meq', 'HCO3_meq', 'Cl_meq', 'SO4_meq'],
        'plot_ilr_classification',
    )
    
    # Classify water types
    df_meq['water_type'] = df_meq.apply(classify_water_type, axis=1)
    df_meq['water_type'] = df_meq['water_type'].astype('category')
    
    # Color palette
    unique_water_types = df_meq['water_type'].cat.categories
    colors = plt.get_cmap('Paired')(np.linspace(0, 1, len(unique_water_types)))
    color_map = dict(zip(unique_water_types, colors))
    
    # Calculate ILR coordinates (guard against zeros/non-positive values)
    ca = df_meq['Ca_meq'].to_numpy(dtype=float)
    mg = df_meq['Mg_meq'].to_numpy(dtype=float)
    na = df_meq['Na_meq'].to_numpy(dtype=float)
    k = df_meq['K_meq'].to_numpy(dtype=float)
    hco3 = df_meq['HCO3_meq'].to_numpy(dtype=float)
    cl = df_meq['Cl_meq'].to_numpy(dtype=float)
    so4 = df_meq['SO4_meq'].to_numpy(dtype=float)

    z1 = np.sqrt(2 / 3) * _safe_log(_safe_divide(np.sqrt(ca * mg), (na + k)))
    z2 = (1 / np.sqrt(2)) * _safe_log(_safe_divide(ca, mg))
    z3 = np.sqrt(2 / 3) * _safe_log(_safe_divide(np.sqrt(cl * so4), hco3))
    z4 = (1 / np.sqrt(2)) * _safe_log(_safe_divide(cl, so4))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7))
    
    sample_colors = [color_map[wt] for wt in df_meq['water_type']]
    
    # Common axis settings to hide -10/10 labels
    ticks = [-7.5, -5, -2.5, 0, 2.5, 5, 7.5]
    
    # Upper Left Panel: Ca/Mg vs Cl/SO4
    ax1.scatter(z2, z4, c=sample_colors, edgecolors='black', s=50, linewidth=0.5)
    ax1.axhline(0, ls='--', color='gray', alpha=0.6)
    ax1.axvline(0, ls='--', color='gray', alpha=0.6)
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_ylabel(r'[SO$_4^{2-}$| Cl$^{-}$]', fontsize=12)
    ax1.set_title(r'[Ca$^{2+}$| Mg$^{2+}$]', fontsize=12, pad=20)
    _add_ilr_arrows(ax1, 'camg_clso4')
    
    # Upper Right Panel: Anions
    ax2.scatter(z3, z4, c=sample_colors, edgecolors='black', s=50, linewidth=0.5)
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    ax2.set_xticks(ticks)
    ax2.set_yticks([])
    ax2.set_title(r'[HCO$_3^{-}$| Cl$^{-}$, SO$_4^{2-}$]', fontsize=12, pad=20)
    ax2.set_ylabel(r'[SO$_4^{2-}$| Cl$^{-}$]', fontsize=12)
    ax2.yaxis.set_label_position('right')
    _add_ilr_field_boundaries(ax2, 'anions')
    
    # Lower Left Panel: Cations
    ax3.scatter(z2, z1, c=sample_colors, edgecolors='black', s=50, linewidth=0.5)
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    ax3.set_xticks(ticks)
    ax3.set_yticks(ticks)
    ax3.set_xlabel(r'[Mg$^{2+}$| Ca$^{2+}$]', fontsize=12)
    ax3.set_ylabel(r'[Ca$^{2+}$, Mg$^{2+}$| Na$^{+}$ + K$^{+}$]', fontsize=12)
    _add_ilr_field_boundaries(ax3, 'cations')
    
    # Lower Right Panel: Diamond
    ax4.scatter(z3, z1, c=sample_colors, edgecolors='black', s=50, linewidth=0.5)
    ax4.axhline(0, ls='--', color='gray', alpha=0.6)
    ax4.axvline(0, ls='--', color='gray', alpha=0.6)
    ax4.set_xlim(-10, 10)
    ax4.set_ylim(-10, 10)
    ax4.set_xticks(ticks)
    ax4.set_yticks([])
    ax4.set_xlabel(r'[HCO$_3^{-}$| Cl$^{-}$, SO$_4^{2-}$]', fontsize=12)
    ax4.set_ylabel(r'[Ca$^{2+}$, Mg$^{2+}$| Na$^{+}$ + K$^{+}$]', fontsize=12)
    ax4.yaxis.set_label_position('right')
    _add_ilr_arrows(ax4, 'diamond')
    
    # Legend with water type percentages
    water_type_counts = df_meq['water_type'].value_counts()
    total_samples = len(df_meq)
    water_type_percentages = (water_type_counts / total_samples * 100).round(1)
    water_type_labels = {wt: f"{wt} ({water_type_percentages.get(wt, 0)}%)" for wt in unique_water_types}
    
    fig.legend(
        handles=[plt.scatter([], [], color=color_map[wt], label=water_type_labels[wt], edgecolors='black') 
                for wt in unique_water_types], 
        loc='center left',
        bbox_to_anchor=(0.85, 0.5),
        fontsize=9,
        frameon=True,
        fancybox=True,
    )
    
    plt.subplots_adjust(wspace=0, hspace=0, right=0.75)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def _add_ilr_arrows(ax: matplotlib.axes.Axes, panel_type: str) -> None:
    """Add ion dominance arrows to ILR panels."""
    if panel_type == 'camg_clso4':
        # X-axis arrows (Ca/Mg)
        ax.arrow(-0.5, 8.5, -1, 0, head_width=0.2, head_length=0.3, fc='black', ec='black')
        ax.text(-4, 9, r'Mg$^{2+}$ $>$ Ca$^{2+}$', fontsize=8, ha='center')
        ax.arrow(0.5, 8.5, 1, 0, head_width=0.2, head_length=0.3, fc='black', ec='black')
        ax.text(4, 9, r'Ca$^{2+}$ $>$ Mg$^{2+}$', fontsize=8, ha='center')
        
        # Y-axis arrows (Cl/SO4)
        ax.arrow(-8.5, -1, 0, -1, head_width=0.2, head_length=0.3, fc='black', ec='black')
        ax.text(-9, -5, r'SO$_4^{2-}$ $>$ Cl$^{-}$', fontsize=8, ha='center', rotation=90)
        ax.arrow(-8.5, 1, 0, 1, head_width=0.2, head_length=0.3, fc='black', ec='black')
        ax.text(-9, 2, r'Cl$^{-}$ $>$ SO$_4^{2-}$', fontsize=8, ha='center', rotation=90)
        
    elif panel_type == 'diamond':
        # X-axis arrows (Anions)
        ax.arrow(-1, 8.5, -1, 0, head_width=0.2, head_length=0.3, fc='black', ec='black')
        ax.text(-5, 9, r'HCO$_3^{-}$ $>$ Cl$^{-}$, SO$_4^{2-}$', fontsize=7, ha='center')
        ax.arrow(1, 8.5, 1, 0, head_width=0.2, head_length=0.3, fc='black', ec='black')
        ax.text(4, 9, r'Cl$^{-}$, SO$_4^{2-}$ $>$ HCO$_3^{-}$', fontsize=7, ha='center')
        
        # Y-axis arrows (Cations)
        ax.arrow(-8.5, -1, 0, -1, head_width=0.2, head_length=0.3, fc='black', ec='black')
        ax.text(-9, -9, r'Ca$^{2+}$, Mg$^{2+}$ $>$ Na$^{+}$ + K$^{+}$', fontsize=7, ha='center', rotation=90)
        ax.arrow(-8.5, 1, 0, 1, head_width=0.2, head_length=0.3, fc='black', ec='black')
        ax.text(-9, 1.5, r'Na$^{+}$ + K$^{+}$ $>$ Ca$^{2+}$, Mg$^{2+}$', fontsize=7, ha='center', rotation=90)


def _add_ilr_field_boundaries(ax: matplotlib.axes.Axes, panel_type: str) -> None:
    """Add field boundary lines to ILR panels."""
    dum1 = np.full(23, 0.5)
    dum2 = np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 
                     4e-1, 4.5e-1, 4.9e-1, 4.99e-1, 4.999e-1, 4.9999e-1, 4.99999e-1, 4.999999e-1, 
                     4.9999999e-1, 4.99999999e-1])
    dum3 = 1 - dum1 - dum2
    
    if panel_type == 'anions':
        # SO4-type field
        ilr1 = np.sqrt(2/3) * np.log(np.sqrt(dum1 * dum2) / dum3)
        ilr2 = np.sqrt(1/2) * np.log(dum2 / dum1)
        ax.plot(ilr1, ilr2, ls='--', color='gray', alpha=0.6)
        ax.text(1, -1.5, r'SO$_4^{2-}$ type', fontsize=10)
        
        # HCO3-type field
        ilr1 = np.sqrt(2/3) * np.log(np.sqrt(dum3 * dum2) / dum1)
        ilr2 = np.sqrt(1/2) * np.log(dum3 / dum2)
        ax.plot(ilr1, ilr2, ls='--', color='gray', alpha=0.6)
        ax.text(-6, -4, r'HCO$_3^{-}$ type', fontsize=10)
        
        # Cl-type field
        ilr1 = np.sqrt(2/3) * np.log(np.sqrt(dum3 * dum1) / dum2)
        ilr2 = np.sqrt(1/2) * np.log(dum1 / dum3)
        ax.plot(ilr1, ilr2, ls='--', color='gray', alpha=0.6)
        ax.text(1, 3, r'Cl$^{-}$ type', fontsize=10)
        
    elif panel_type == 'cations':
        # Mg-type field
        ilr1 = np.sqrt(2/3) * np.log(np.sqrt(dum1 * dum2) / dum3)
        ilr2 = np.sqrt(1/2) * np.log(dum2 / dum1)
        ax.plot(ilr2, ilr1, ls='--', color='gray', alpha=0.6)
        ax.text(-4, 4, r'Mg$^{2+}$ type', fontsize=10)
        
        # Na+K-type field
        ilr1 = np.sqrt(2/3) * np.log(np.sqrt(dum3 * dum2) / dum1)
        ilr2 = np.sqrt(1/2) * np.log(dum3 / dum2)
        ax.plot(ilr2, ilr1, ls='--', color='gray', alpha=0.6)
        ax.text(2, -5, r'Ca$^{2+}$ type', fontsize=10)
        
        # Ca-type field
        ilr1 = np.sqrt(2/3) * np.log(np.sqrt(dum3 * dum1) / dum2)
        ilr2 = np.sqrt(1/2) * np.log(dum1 / dum3)
        ax.plot(ilr2, ilr1, ls='--', color='gray', alpha=0.6)
        ax.text(2, 0.5, r'Na$^{+}$+K$^{+}$ type', fontsize=10)


def _plot_ilr_classification(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    df_meq: pd.DataFrame,
    **kwargs
) -> None:
    """Wrapper for registry - ILR is a multi-panel figure."""
    # This is handled specially in create_figure
    pass


# =============================================================================
# CORRELATION MATRIX
# =============================================================================

def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Create lower-triangle correlation heatmap with annotations.
    
    Parameters
    ----------
    df : DataFrame
        Input data
    columns : list, optional
        Columns to include. If None, uses all numeric columns.
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()
    
    if columns is None:
        df_numeric = df.select_dtypes(include=[np.number])
    else:
        df_numeric = df[columns]
    
    corr = df_numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title('Ion Correlation Matrix', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def _plot_correlation_matrix(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    df_meq: pd.DataFrame,
    **kwargs
) -> None:
    """Render correlation matrix onto axes (for registry)."""
    df_numeric = df.select_dtypes(include=[np.number])
    corr = df_numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title('Ion Correlation Matrix', fontsize=12)


# =============================================================================
# MINERAL PLOTS
# =============================================================================

def plot_mineral_fractions(
    mineral_result,
    sample_indices: Optional[List[int]] = None,
    output_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Stacked bar chart of mineral contributions per sample.
    
    Parameters
    ----------
    mineral_result : MineralInversionResult
        Result from mineral inversion
    sample_indices : list, optional
        Indices of samples to plot. If None, plots first 20.
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()
    
    fractions = mineral_result.mineral_fractions
    names = mineral_result.mineral_names or [f'M{i}' for i in range(fractions.shape[1])]
    
    if sample_indices is None:
        sample_indices = list(range(min(20, fractions.shape[0])))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sample_indices))
    fractions_sel = fractions[sample_indices, :]
    
    # Filter for active minerals (contribution > 1% in at least one sample)
    active_mask = np.any(fractions_sel > 0.01, axis=0)
    names_active = [names[i] for i in range(len(names)) if active_mask[i]]
    fractions_active = fractions_sel[:, active_mask]
    
    if len(names_active) == 0:
        ax.text(0.5, 0.5, 'No significant mineral contributions', ha='center', va='center')
        return fig
    
    bottom = np.zeros(len(sample_indices))

    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(names_active)))
    
    for i, (name, color) in enumerate(zip(names_active, colors)):
        values = fractions_active[:, i]
        ax.bar(x, values, bottom=bottom, label=name, color=color, edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Mineral Fraction', fontsize=12)
    ax.set_title('Mineral Composition', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in sample_indices], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def _plot_mineral_fractions(
    ax: matplotlib.axes.Axes,
    mineral_result,
    **kwargs
) -> None:
    """Render mineral fractions onto axes (for registry)."""
    if mineral_result is None:
        ax.text(0.5, 0.5, 'No mineral data', ha='center', va='center', transform=ax.transAxes)
        return
    
    fractions = mineral_result.mineral_fractions
    names = mineral_result.mineral_names or [f'M{i}' for i in range(fractions.shape[1])]
    
    sample_indices = list(range(min(10, fractions.shape[0])))
    fractions_sel = fractions[sample_indices, :]
    
    # Filter for active minerals (contribution > 1% in at least one sample)
    active_mask = np.any(fractions_sel > 0.01, axis=0)
    names_active = [names[i] for i in range(len(names)) if active_mask[i]]
    fractions_active = fractions_sel[:, active_mask]
    
    if len(names_active) == 0:
        ax.text(0.5, 0.5, 'No significant contributions', ha='center', va='center', transform=ax.transAxes)
        return

    x = np.arange(len(sample_indices))
    bottom = np.zeros(len(sample_indices))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(names_active)))
    
    for i, (name, color) in enumerate(zip(names_active, colors)):
        values = fractions_active[:, i]
        ax.bar(x, values, bottom=bottom, label=name, color=color, edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_xlabel('Sample', fontsize=10)
    ax.set_ylabel('Fraction', fontsize=10)
    ax.set_title('Mineral Fractions', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in sample_indices], rotation=45, ha='right', fontsize=8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.set_ylim(0, 1)


def plot_saturation_indices(
    mineral_result,
    sample_index: int = 0,
    output_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Horizontal bar chart showing SI values for each mineral.
    
    Parameters
    ----------
    mineral_result : MineralInversionResult
        Result from mineral inversion with saturation_indices
    sample_index : int
        Sample to plot
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()
    
    if mineral_result.saturation_indices is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No saturation index data available', ha='center', va='center')
        return fig
    
    si_dict = mineral_result.saturation_indices
    names = list(si_dict.keys())
    values = [si_dict[name][sample_index] for name in names]
    
    # Filter out -999 (missing data)
    valid = [(n, v) for n, v in zip(names, values) if v > -900]
    if not valid:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No valid SI data for this sample', ha='center', va='center')
        return fig
    
    names, values = zip(*valid)
    values = np.array(values)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.3)))
    
    colors = ['#2ecc71' if v < 0 else '#e74c3c' for v in values]
    
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(0.5, color='orange', linewidth=1, linestyle='--', label='Supersaturation threshold')
    
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Saturation Index (SI)', fontsize=12)
    ax.set_title(f'Mineral Saturation Indices (Sample {sample_index})', fontsize=14)
    ax.legend(loc='best')
    
    # Color legend
    ax.text(0.02, 0.98, '● Undersaturated (can dissolve)', color='#2ecc71', 
            transform=ax.transAxes, fontsize=9, va='top')
    ax.text(0.02, 0.93, '● Supersaturated (precipitation)', color='#e74c3c', 
            transform=ax.transAxes, fontsize=9, va='top')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def _plot_saturation_indices(
    ax: matplotlib.axes.Axes,
    mineral_result,
    **kwargs
) -> None:
    """Render SI bar chart onto axes (for registry)."""
    if mineral_result is None or mineral_result.saturation_indices is None:
        ax.text(0.5, 0.5, 'No SI data', ha='center', va='center', transform=ax.transAxes)
        return
    
    si_dict = mineral_result.saturation_indices
    names = list(si_dict.keys())
    values = [si_dict[name][0] for name in names]
    
    valid = [(n, v) for n, v in zip(names, values) if v > -900]
    if not valid:
        ax.text(0.5, 0.5, 'No valid SI', ha='center', va='center', transform=ax.transAxes)
        return
    
    names, values = zip(*valid)
    values = np.array(values)
    colors = ['#2ecc71' if v < 0 else '#e74c3c' for v in values]
    
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('SI', fontsize=10)
    ax.set_title('Saturation Indices', fontsize=11)


def _plot_mineral_plausibility(
    ax: matplotlib.axes.Axes,
    mineral_result,
    **kwargs
) -> None:
    """Render thermodynamic plausibility heatmap onto axes."""
    if mineral_result is None or mineral_result.thermo_plausible is None:
        ax.text(0.5, 0.5, 'No thermo data', ha='center', va='center', transform=ax.transAxes)
        return
    
    data = mineral_result.thermo_plausible.astype(int)[:10, :]  # First 10 samples
    names = mineral_result.mineral_names or [f'M{i}' for i in range(data.shape[1])]
    
    sns.heatmap(
        data.T,
        cmap='RdYlGn',
        center=0.5,
        ax=ax,
        cbar=False,
        yticklabels=names,
        xticklabels=[str(i) for i in range(data.shape[0])],
    )
    ax.set_xlabel('Sample', fontsize=10)
    ax.set_title('Thermo Plausibility', fontsize=11)


# =============================================================================
# VIP PLOTS
# =============================================================================

def plot_vip_decomposition(
    nvip_result,
    output_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Stacked bar chart showing VIP_T, VIP_I, VIP_F contributions per ion.
    
    Parameters
    ----------
    nvip_result : DataFrame or object with VIP_T, VIP_I, VIP_F columns
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()
    
    if isinstance(nvip_result, pd.DataFrame):
        df = nvip_result.copy()
    else:
        # Assume it has attributes
        df = pd.DataFrame({
            'feature': nvip_result.feature_names,
            'VIP_T': nvip_result.VIP_T,
            'VIP_I': nvip_result.VIP_I,
            'VIP_F': nvip_result.VIP_F,
        })
    
    df = df.sort_values('VIP_T', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    
    ax.bar(x, df['VIP_T'], label='VIP_T (Truth)', color='#3498db')
    ax.bar(x, df['VIP_I'], bottom=df['VIP_T'], label='VIP_I (Indeterminacy)', color='#9b59b6')
    ax.bar(x, df['VIP_F'], bottom=df['VIP_T'] + df['VIP_I'], label='VIP_F (Falsity)', color='#e74c3c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['feature'], rotation=45, ha='right')
    ax.set_ylabel('VIP (stacked)', fontsize=12)
    ax.set_title('NVIP Channel Decomposition', fontsize=14)
    ax.legend()
    ax.axhline(1, color='gray', linestyle='--', alpha=0.7, label='Importance threshold')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def _plot_vip_decomposition(
    ax: matplotlib.axes.Axes,
    nvip_result,
    **kwargs
) -> None:
    """Render VIP decomposition onto axes (for registry)."""
    if nvip_result is None:
        ax.text(0.5, 0.5, 'No VIP data', ha='center', va='center', transform=ax.transAxes)
        return
    
    if isinstance(nvip_result, pd.DataFrame):
        df = nvip_result.copy()
    else:
        df = pd.DataFrame({
            'feature': getattr(nvip_result, 'feature_names', []),
            'VIP_T': getattr(nvip_result, 'VIP_T', []),
            'VIP_I': getattr(nvip_result, 'VIP_I', []),
            'VIP_F': getattr(nvip_result, 'VIP_F', []),
        })
    
    if df.empty:
        ax.text(0.5, 0.5, 'No VIP data', ha='center', va='center', transform=ax.transAxes)
        return
    
    df = df.sort_values('VIP_T', ascending=False)
    x = np.arange(len(df))
    
    ax.bar(x, df['VIP_T'], label='T', color='#3498db')
    ax.bar(x, df['VIP_I'], bottom=df['VIP_T'], label='I', color='#9b59b6')
    ax.bar(x, df['VIP_F'], bottom=df['VIP_T'] + df['VIP_I'], label='F', color='#e74c3c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['feature'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('VIP', fontsize=10)
    ax.set_title('VIP Decomposition', fontsize=11)
    ax.legend(fontsize=8)


def _plot_vip_aggregate(
    ax: matplotlib.axes.Axes,
    nvip_result,
    **kwargs
) -> None:
    """Render VIP aggregate bar chart onto axes."""
    if nvip_result is None:
        ax.text(0.5, 0.5, 'No VIP data', ha='center', va='center', transform=ax.transAxes)
        return
    
    if isinstance(nvip_result, pd.DataFrame):
        df = nvip_result.copy()
        if 'VIP_agg' not in df.columns:
            df['VIP_agg'] = np.sqrt(df['VIP_T']**2 + df['VIP_I']**2 + df['VIP_F']**2)
    else:
        df = pd.DataFrame({
            'feature': getattr(nvip_result, 'feature_names', []),
            'VIP_agg': getattr(nvip_result, 'VIP_agg', []),
        })
    
    if df.empty:
        ax.text(0.5, 0.5, 'No VIP data', ha='center', va='center', transform=ax.transAxes)
        return
    
    df = df.sort_values('VIP_agg', ascending=False)
    
    sns.barplot(x='feature', y='VIP_agg', data=df, palette='viridis', ax=ax, hue='feature')
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.axhline(1, color='red', linestyle='--', alpha=0.7)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df['feature'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('VIP_agg', fontsize=10)
    ax.set_xlabel('')
    ax.set_title('VIP Aggregate', fontsize=11)


def _plot_baseline_fraction(
    ax: matplotlib.axes.Axes,
    nvip_result,
    **kwargs
) -> None:
    """Render baseline fraction (pi_G) bar chart onto axes."""
    if nvip_result is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    if isinstance(nvip_result, pd.DataFrame):
        if 'pi_G' in nvip_result.columns:
            df = nvip_result[['feature', 'pi_G']].copy() if 'feature' in nvip_result.columns else nvip_result
        else:
            ax.text(0.5, 0.5, 'No pi_G data', ha='center', va='center', transform=ax.transAxes)
            return
    else:
        pi_G = getattr(nvip_result, 'pi_G', [])
        names = getattr(nvip_result, 'feature_names', getattr(nvip_result, 'ion_names', kwargs.get('feature_names', [])))
        
        # If names still missing but we have pi_G, use numeric indices
        if len(names) == 0 and len(pi_G) > 0:
            names = [f'Ion {i}' for i in range(len(pi_G))]
            
        df = pd.DataFrame({
            'ion': names,
            'pi_G': pi_G,
        })
    
    if df.empty or 'pi_G' not in df.columns:
        ax.text(0.5, 0.5, 'No pi_G data', ha='center', va='center', transform=ax.transAxes)
        return
    
    df = df.sort_values('pi_G', ascending=False)
    ion_col = 'ion' if 'ion' in df.columns else 'feature'
    
    sns.barplot(x=ion_col, y='pi_G', data=df, palette='magma', ax=ax, hue=ion_col)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df[ion_col], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('πG', fontsize=10)
    ax.set_xlabel('')
    ax.set_title('Baseline Fraction (πG)', fontsize=11)


def _plot_g_histogram(
    ax: matplotlib.axes.Axes,
    samples_result,
    **kwargs
) -> None:
    """Render sample G distribution histogram onto axes."""
    if samples_result is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    if isinstance(samples_result, pd.DataFrame) and 'G' in samples_result.columns:
        g_values = samples_result['G']
    elif hasattr(samples_result, 'G'):
        g_values = samples_result.G
    else:
        ax.text(0.5, 0.5, 'No G data', ha='center', va='center', transform=ax.transAxes)
        return
    
    sns.histplot(x=g_values, bins=20, kde=False, color='steelblue', ax=ax)
    ax.set_xlabel('G (baseline fraction)', fontsize=10)
    ax.set_title('Sample G Distribution', fontsize=11)


# =============================================================================
# MODEL DIAGNOSTIC PLOTS
# =============================================================================

def _plot_pls_loadings(
    ax: matplotlib.axes.Axes,
    nvip_result,
    **kwargs
) -> None:
    """Render PLS loadings (partitioned by channel) onto axes."""
    if nvip_result is None or not hasattr(nvip_result, 'model'):
        ax.text(0.5, 0.5, 'No model data', ha='center', va='center', transform=ax.transAxes)
        return
    
    model = nvip_result.model
    if model.components_ is None:
        ax.text(0.5, 0.5, 'Model not fitted', ha='center', va='center', transform=ax.transAxes)
        return
        
    p = model.components_.W.shape[0] // 3
    feature_names = nvip_result.feature_names or [f'X{i}' for i in range(p)]
    
    # Take first component loadings
    w1 = model.components_.W[:, 0]
    w_t = w1[:p]
    w_i = w1[p:2*p]
    w_f = w1[2*p:]
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    ax.bar(x - width, w_t, width, label='T', color='#3498db')
    ax.bar(x, w_i, width, label='I', color='#9b59b6')
    ax.bar(x + width, w_f, width, label='F', color='#e74c3c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Weight (Comp 1)', fontsize=10)
    ax.set_title('PLS Loadings by Channel', fontsize=11)
    ax.legend(fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)


def _plot_explained_variance(
    ax: matplotlib.axes.Axes,
    nvip_result,
    **kwargs
) -> None:
    """Render explained variance scree plot onto axes."""
    if nvip_result is None or not hasattr(nvip_result, 'model'):
        ax.text(0.5, 0.5, 'No model data', ha='center', va='center', transform=ax.transAxes)
        return
    
    model = nvip_result.model
    if model.components_ is None:
        ax.text(0.5, 0.5, 'Model not fitted', ha='center', va='center', transform=ax.transAxes)
        return
        
    ev = model.components_.explained_variance
    x = np.arange(len(ev)) + 1
    
    ax.bar(x, ev, color='steelblue', alpha=0.7)
    ax.plot(x, np.cumsum(ev), marker='o', color='darkorange', label='Cumulative')
    
    ax.set_xticks(x)
    ax.set_xlabel('Component', fontsize=10)
    ax.set_ylabel('Variance Explained', fontsize=10)
    ax.set_title('Scree Plot', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)


# =============================================================================
# LAYOUT PARSER AND FIGURE GENERATOR
# =============================================================================

def parse_layout(layout_str: str) -> List[List[str]]:
    """
    Parse grid layout string into 2D list of plot codes.
    
    Example: '[g1|g2][i1]' -> [['g1', 'g2'], ['i1']]
    """
    if layout_str in PRESETS:
        layout_str = PRESETS[layout_str]
    
    rows = re.findall(r'\[([^\]]+)\]', layout_str)
    grid = [row.split('|') for row in rows]
    return grid


def create_figure(
    layout: str,
    data: Dict[str, Any],
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Create a composite figure from layout specification.
    
    Parameters
    ----------
    layout : str
        Layout string like '[g1|g2][i1]' or preset name like 'gibbs'
    data : dict
        Data dictionary with keys: 'df', 'df_meq', 'mineral_result', 'nvip_result'
    figsize : tuple, optional
        Override figure size
    output_path : str, optional
        Save path
    
    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()
    
    grid = parse_layout(layout)
    n_rows = len(grid)
    n_cols = max(len(row) for row in grid)
    
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    df = data.get('df')
    df_meq = data.get('df_meq')
    if df_meq is None and df is not None:
        df_meq = mg_to_meq(df)
    
    mineral_result = data.get('mineral_result')
    nvip_result = data.get('nvip_result')
    nsr_result = data.get('nsr_result')
    samples_result = data.get('samples_result')
    
    for i, row in enumerate(grid):
        for j, code in enumerate(row):
            ax = axes[i, j]
            
            if code not in PLOT_REGISTRY:
                ax.text(0.5, 0.5, f'Unknown: {code}', ha='center', va='center', transform=ax.transAxes)
                continue
            
            plot_info = PLOT_REGISTRY[code]
            func_name = plot_info['func']
            plot_type = plot_info['type']
            
            # Special handling for multi-panel plots
            if code == 'i1':
                # ILR is a 2x2 figure - we'll put a placeholder
                ax.text(0.5, 0.5, 'ILR: Use plot_ilr_classification()', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                continue
            
            # Get the function
            func = globals().get(func_name)
            if func is None:
                ax.text(0.5, 0.5, f'Missing: {func_name}', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Call with appropriate data
            if plot_type == 'gibbs':
                if df is None or df_meq is None:
                    ax.text(0.5, 0.5, 'Missing df/df_meq', ha='center', va='center', transform=ax.transAxes)
                else:
                    func(ax, df, df_meq)
            elif plot_type == 'diagnostic':
                if df is None:
                    ax.text(0.5, 0.5, 'Missing df', ha='center', va='center', transform=ax.transAxes)
                else:
                    func(ax, df, df_meq)
            elif plot_type == 'mineral':
                func(ax, mineral_result)
            elif plot_type == 'vip' or plot_type == 'model':
                if 'g_histogram' in func_name:
                    func(ax, samples_result)
                elif 'baseline_fraction' in func_name:
                    func(ax, nsr_result or nvip_result, feature_names=getattr(nvip_result, 'feature_names', None)) 
                else:
                    func(ax, nvip_result)
    
    # Hide unused axes
    for i in range(len(grid), n_rows):
        for j in range(n_cols):
            axes[i, j].set_visible(False)
    for i, row in enumerate(grid):
        for j in range(len(row), n_cols):
            axes[i, j].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


# =============================================================================
# MASTER REPORT GENERATOR
# =============================================================================

def generate_report(
    df: pd.DataFrame,
    output_dir: str = 'figures',
    preset: str = 'core',
    mineral_result = None,
    nvip_result = None,
    samples_result = None
) -> None:
    """
    Generate all standard hydrogeochemical plots with a single call.
    
    Parameters
    ----------
    df : DataFrame
        Raw ion data in mg/L with columns: Ca, Mg, Na, K, HCO3, Cl, SO4, NO3, F, TDS
    output_dir : str
        Output directory for saved figures
    preset : str
        Layout preset: 'gibbs', 'ilr', 'hydrochem', 'minerals', 'vip', 'core', 'full'
    mineral_result : MineralInversionResult, optional
        Results from mineral inversion
    nvip_result : optional
        Results from NVIP analysis
    samples_result : optional
        Sample-level results with G column
    
    Example
    -------
    >>> from neutrohydro.visualization import generate_report
    >>> import pandas as pd
    >>> df = pd.read_csv("data.csv")
    >>> generate_report(df, output_dir="my_figures", preset="hydrochem")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df_meq = mg_to_meq(df)
    df_meq = calculate_ion_sums(df_meq)
    
    print(f"Ionic Balance Summary:")
    print(f"  Mean error: {df_meq['Ion_Balance_Error'].mean():.2f}%")
    print(f"  Acceptable (<10%): {(abs(df_meq['Ion_Balance_Error']) < 10).sum()}/{len(df_meq)}")
    
    # Always generate individual plots
    print(f"\nGenerating plots to {output_dir}/...")
    
    # Gibbs
    plot_gibbs(df, df_meq, output_path=os.path.join(output_dir, 'gibbs_plot.png'))
    print("  [OK] gibbs_plot.png")
    
    # ILR
    plot_ilr_classification(df, df_meq, output_path=os.path.join(output_dir, 'ilr_classification.png'))
    print("  [OK] ilr_classification.png")
    
    # Correlation
    plot_correlation_matrix(df, output_path=os.path.join(output_dir, 'correlation_matrix.png'))
    print("  [OK] correlation_matrix.png")
    
    # Mineral plots (if data available)
    if mineral_result is not None:
        plot_mineral_fractions(mineral_result, output_path=os.path.join(output_dir, 'mineral_fractions.png'))
        print("  [OK] mineral_fractions.png")
        
        if mineral_result.saturation_indices is not None:
            plot_saturation_indices(mineral_result, output_path=os.path.join(output_dir, 'saturation_indices.png'))
            print("  [OK] saturation_indices.png")
    
    # VIP plots (if data available)
    if nvip_result is not None:
        plot_vip_decomposition(nvip_result, output_path=os.path.join(output_dir, 'vip_decomposition.png'))
        print("  [OK] vip_decomposition.png")
    
    print(f"\n[OK] Report complete! All plots saved to {output_dir}/")


# =============================================================================
# JACOBIAN-ORTHOGONAL ATTRIBUTION PLOTS
# =============================================================================
#
# One fixed channel -> color mapping for every orthogonal-attribution plot
# (first three slots of the validated reference categorical palette; all
# pairs pass the CVD and normal-vision floors). The aqua I slot sits below 3:1
# contrast on white, so every plot carries a legend and/or direct labels.

CHANNEL_COLORS = {
    'T': '#2a78d6',  # baseline (blue)
    'F': '#eb6834',  # perturbation (orange)
    'I': '#1baf7a',  # indeterminate (aqua)
}
CHANNEL_LABELS = {
    'T': 'T (baseline)',
    'I': 'I (indeterminate)',
    'F': 'F (perturbation)',
}
_UNEXPLAINED_COLOR = '#d6d5d0'   # neutral gray for "not explained"
_TEXT_PRIMARY = '#0b0b0b'
_TEXT_SECONDARY = '#52514e'
_TEXT_MUTED = '#8a8984'
_SURFACE = '#ffffff'


def _pretty_ion(name: str) -> str:
    """Typeset an ion name ('Ca2+', 'HCO3-', 'Ca') with ION_MAP when possible."""
    if name in ION_MAP:
        return ION_MAP[name]
    for suffix in ('2+', '2-', '+', '-'):
        if name.endswith(suffix) and name[:-len(suffix)] in ION_MAP:
            return ION_MAP[name[:-len(suffix)]]
    return name


def _get_axes(ax, figsize):
    """Return (fig, ax, own_figure)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


def _finish(fig, output_path: Optional[str], own_figure: bool):
    """Shared save/close convention of this module."""
    if own_figure:
        fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if own_figure:
            plt.close(fig)
    return fig


def _recessive_axes(ax, grid_axis: str = 'x') -> None:
    """Thin, quiet axes: no top/right spines, light grid on one axis."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(_TEXT_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_TEXT_SECONDARY, length=3)
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color='#e4e3df', linewidth=0.8)
        ax.set_axisbelow(True)


def _ternary_xy(tau, iota, phi):
    """Barycentric (T bottom-left, F bottom-right, I top) -> Cartesian."""
    tau = np.asarray(tau, dtype=float)
    iota = np.asarray(iota, dtype=float)
    phi = np.asarray(phi, dtype=float)
    total = tau + iota + phi
    total = np.where(total > 0, total, 1.0)
    iota_n, phi_n = iota / total, phi / total
    return phi_n + 0.5 * iota_n, (np.sqrt(3) / 2) * iota_n


def plot_neutrosophic_ternary(
    attribution=None,
    tau: Optional[np.ndarray] = None,
    iota: Optional[np.ndarray] = None,
    phi: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    color_by: Optional[np.ndarray] = None,
    color_label: str = '',
    max_labels: int = 15,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = 'Neutrosophic memberships per sample',
    output_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Ternary diagram of per-sample neutrosophic memberships (tau, iota, phi).

    With the Jacobian-orthogonal encoding these are the exact energy
    fractions of each sample's deviation from the center lying in the
    baseline (T), indeterminate (I) and perturbation (F) subspaces, so
    they sum to 1 and every sample is a point in the triangle. Corners:
    T bottom-left, F bottom-right, I top; the horizontal position reads
    baseline -> perturbation.

    Parameters
    ----------
    attribution : OrthogonalAttribution, optional
        Source of ``tau``, ``iota``, ``phi`` (ignored for any of them
        passed explicitly).
    tau, iota, phi : array-like of shape (n,), optional
        Memberships (normalised to sum to 1 if they do not already).
    labels : list of str, optional
        Sample labels; the ``max_labels`` most perturbed samples (largest
        phi) are annotated.
    color_by : array-like of shape (n,), optional
        Numeric values (e.g. TDS, depth) shown with a single-hue sequential
        colormap and colorbar. By default points are colored by their
        dominant channel.
    color_label : str
        Colorbar label when ``color_by`` is given.
    max_labels : int, default=15
        Maximum number of annotated samples.
    ax : matplotlib Axes, optional
        Axes to draw on (a new figure is created otherwise).
    title : str, optional
    output_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()

    if attribution is not None:
        tau = attribution.tau if tau is None else tau
        iota = attribution.iota if iota is None else iota
        phi = attribution.phi if phi is None else phi
    if tau is None or iota is None or phi is None:
        raise ValueError("Provide an OrthogonalAttribution or tau, iota and phi")
    tau = np.asarray(tau, dtype=float)
    iota = np.asarray(iota, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if not (tau.shape == iota.shape == phi.shape):
        raise ValueError("tau, iota and phi must have the same shape")

    fig, ax, own = _get_axes(ax, (7.6, 6.4))
    h = np.sqrt(3) / 2

    # Gridlines at 0.2 steps for each membership, with tick labels on the edge
    # each membership is read from: phi (bottom), iota (right), tau (left)
    tick = dict(fontsize=8, color=_TEXT_MUTED)
    for v in (0.2, 0.4, 0.6, 0.8):
        for a, b in (
            ((v, 1 - v, 0.0), (v, 0.0, 1 - v)),   # tau = v
            ((1 - v, v, 0.0), (0.0, v, 1 - v)),   # iota = v
            ((1 - v, 0.0, v), (0.0, 1 - v, v)),   # phi = v
        ):
            x, y = _ternary_xy([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
            ax.plot(x, y, color='#e4e3df', linewidth=0.8, zorder=1)
        x, y = _ternary_xy(1 - v, 0.0, v)
        ax.text(x, y - 0.03, f'{v:.1f}', ha='center', va='top', **tick)
        x, y = _ternary_xy(0.0, v, 1 - v)
        ax.text(x + 0.025, y, f'{v:.1f}', ha='left', va='center', **tick)
        x, y = _ternary_xy(v, 1 - v, 0.0)
        ax.text(x - 0.025, y, f'{v:.1f}', ha='right', va='center', **tick)

    # Triangle outline
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color=_TEXT_SECONDARY, linewidth=1.0, zorder=2)

    # Corner labels (inside the axes limits so nothing overflows)
    corner = dict(fontsize=11, color=_TEXT_PRIMARY)
    ax.text(-0.03, -0.03, CHANNEL_LABELS['T'], ha='right', va='top', **corner)
    ax.text(1.03, -0.03, CHANNEL_LABELS['F'], ha='left', va='top', **corner)
    ax.text(0.5, h + 0.04, CHANNEL_LABELS['I'], ha='center', va='bottom', **corner)

    # Edge titles: which membership the tick labels on each edge measure
    edge = dict(fontsize=9, color=_TEXT_SECONDARY, ha='center', va='center')
    ax.text(0.5, -0.13, r'$\phi$  perturbation share $\rightarrow$', **edge)
    ax.text(0.25 - 0.13 * np.cos(np.pi / 6), h / 2 + 0.13 * 0.5,
            r'$\leftarrow$ $\tau$  baseline share', rotation=60, **edge)
    ax.text(0.75 + 0.13 * np.cos(np.pi / 6), h / 2 + 0.13 * 0.5,
            r'$\iota$  indeterminate share $\rightarrow$', rotation=-60, **edge)

    xs, ys = _ternary_xy(tau, iota, phi)
    marker = dict(s=56, edgecolors=_SURFACE, linewidths=1.2, zorder=3)
    if color_by is not None:
        from matplotlib.colors import LinearSegmentedColormap

        values = np.asarray(color_by, dtype=float)
        if values.shape != tau.shape:
            raise ValueError("color_by must have one value per sample")
        # Single-hue sequential ramp, truncated so the lowest values do not
        # vanish into the white surface
        cmap = LinearSegmentedColormap.from_list(
            'blues_truncated', plt.get_cmap('Blues')(np.linspace(0.3, 1.0, 256))
        )
        sc = ax.scatter(xs, ys, c=values, cmap=cmap, **marker)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
        cbar.set_label(color_label, color=_TEXT_SECONDARY)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=_TEXT_SECONDARY, length=2)
    else:
        dominant = np.argmax(np.column_stack([tau, iota, phi]), axis=1)
        for k, ch in enumerate(('T', 'I', 'F')):
            mask = dominant == k
            ax.scatter(xs[mask], ys[mask], color=CHANNEL_COLORS[ch],
                       label=f'dominant {ch} (n={int(mask.sum())})', **marker)
        ax.legend(loc='upper left', frameon=False, fontsize=9, title='Samples',
                  title_fontsize=9)

    if labels is not None:
        labels = list(labels)
        if len(labels) != len(tau):
            raise ValueError("labels must have one entry per sample")
        order = np.argsort(-phi)[:max(0, max_labels)]
        # cycle label positions around the point to reduce collisions
        offsets = ((6, 5), (-6, 5), (6, -11), (-6, -11))
        for rank, i in enumerate(order):
            dx, dy = offsets[rank % len(offsets)]
            ax.annotate(str(labels[i]), (xs[i], ys[i]), xytext=(dx, dy), textcoords='offset points',
                        ha='left' if dx > 0 else 'right', fontsize=8, color=_TEXT_SECONDARY,
                        zorder=4)

    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(-0.2, h + 0.14)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12, color=_TEXT_PRIMARY)

    return _finish(fig, output_path, own)


def plot_exact_r2_split(
    attribution,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Single stacked bar of the exact R^2 split R^2 = R^2_T + R^2_I + R^2_F.

    The unexplained remainder 1 - R^2 is shown in neutral gray. With the
    Jacobian-orthogonal encoding the channel predictions are mutually
    orthogonal, so the three parts add up to R^2 exactly (no interaction
    term and no order dependence).

    Parameters
    ----------
    attribution : OrthogonalAttribution
        Must have been computed with the target (``R2_channels`` set).
    ax : matplotlib Axes, optional
    title : str, optional
        Defaults to a title stating the split.
    output_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()

    parts = getattr(attribution, 'R2_channels', None)
    if parts is None:
        raise ValueError("attribution has no R2_channels; pass y to orthogonal_attribution")
    parts = np.clip(np.asarray(parts, dtype=float), 0.0, None)
    r2 = float(parts.sum())
    reported = attribution.R2 if getattr(attribution, 'R2', None) is not None else r2
    remainder = max(0.0, 1.0 - r2)

    fig, ax, own = _get_axes(ax, (8.5, 2.6))
    left = 0.0
    segments = [(CHANNEL_LABELS[c], parts[k], CHANNEL_COLORS[c]) for k, c in enumerate('TIF')]
    segments.append(('not explained', remainder, _UNEXPLAINED_COLOR))
    for name, width, color in segments:
        ax.barh(0, width, left=left, height=0.5, color=color, edgecolor=_SURFACE,
                linewidth=2, label=f'{name}: {width:.3f}')
        if width >= 0.06:
            ax.text(left + width / 2, 0, f'{width:.2f}', ha='center', va='center', fontsize=10,
                    color=_TEXT_PRIMARY if color in (_UNEXPLAINED_COLOR, CHANNEL_COLORS['I']) else _SURFACE,
                    fontweight='bold')
        left += width

    ax.set_xlim(0, max(1.0, left))
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel(r'Share of target variance ($R^2$)', color=_TEXT_SECONDARY)
    _recessive_axes(ax, grid_axis='')
    ax.spines['left'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0), borderaxespad=4.2,
              ncol=4, frameon=False, fontsize=9)
    if title is None:
        title = (rf'Exact $R^2$ split: $R^2$ = {reported:.3f} = '
                 rf'T {parts[0]:.3f} + I {parts[1]:.3f} + F {parts[2]:.3f}')
    ax.set_title(title, fontsize=11, color=_TEXT_PRIMARY, loc='left')

    return _finish(fig, output_path, own)


def plot_ion_channel_shares(
    attribution,
    feature_names: Optional[List[str]] = None,
    gamma: Optional[float] = 0.7,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = 'Exact channel shares of each ion\'s predictive energy',
    output_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Stacked horizontal bars of pi_T, pi_I, pi_F per ion (each row sums to 1).

    Ions are sorted by pi_T (most baseline-carried at the top). Optional
    dashed guides mark the classification thresholds gamma and 1 - gamma.

    Parameters
    ----------
    attribution : OrthogonalAttribution
    feature_names : list of str, optional
        Defaults to ``attribution.feature_names``.
    gamma : float or None, default=0.7
        Classification threshold to draw (None to omit).
    ax : matplotlib Axes, optional
    title : str, optional
    output_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()

    pi = np.column_stack([attribution.pi_T, attribution.pi_I, attribution.pi_F]).astype(float)
    p = pi.shape[0]
    names = feature_names or getattr(attribution, 'feature_names', None) or [f'Ion_{j}' for j in range(p)]
    if len(names) != p:
        raise ValueError("feature_names must have one entry per ion")

    order = np.argsort(pi[:, 0])  # ascending -> largest pi_T drawn at the top
    fig, ax, own = _get_axes(ax, (7.5, 0.42 * p + 1.6))
    y = np.arange(p)
    left = np.zeros(p)
    for k, ch in enumerate('TIF'):
        width = pi[order, k]
        ax.barh(y, width, left=left, height=0.62, color=CHANNEL_COLORS[ch], edgecolor=_SURFACE,
                linewidth=1.5, label=CHANNEL_LABELS[ch])
        left += width

    # Direct label: the baseline share at the end of the T segment, when it fits
    for row, j in enumerate(order):
        if pi[j, 0] >= 0.12:
            ax.text(pi[j, 0] - 0.01, row, f'{pi[j, 0]:.2f}', ha='right', va='center',
                    fontsize=8, color=_SURFACE, fontweight='bold')

    if gamma is not None:
        for g, lab in ((gamma, r'$\gamma$'), (1 - gamma, r'$1-\gamma$')):
            ax.axvline(g, color=_TEXT_MUTED, linestyle='--', linewidth=0.8, zorder=3)
            ax.text(g, 1.0, lab, transform=ax.get_xaxis_transform(), ha='center',
                    va='bottom', fontsize=8, color=_TEXT_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_ion(names[j]) for j in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Share of predictive energy', color=_TEXT_SECONDARY)
    _recessive_axes(ax, grid_axis='')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0), borderaxespad=4.0,
              ncol=3, frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontsize=11, color=_TEXT_PRIMARY, loc='left', pad=16)

    return _finish(fig, output_path, own)


def plot_process_alignment(
    alignment_df: pd.DataFrame,
    metric: str = 'T_vs_F',
    eps: float = 0.01,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Diverging bars: is each process baseline- or perturbation-aligned?

    Takes the DataFrame of :func:`neutrohydro.orthogonal.subspace_alignment`.
    With ``metric="T_vs_F"`` (default) each bar is
    ``log2((enrich_T + eps) / (enrich_F + eps))``: positive means the
    process direction is preferentially aligned with the baseline subspace,
    negative with the perturbation subspace, 0 means no preference
    relative to a random direction. ``metric="T"`` plots
    ``log2(enrich_T + eps)`` (alignment with the baseline vs chance).
    Processes whose dominant channel is I are tagged "(I)".

    Parameters
    ----------
    alignment_df : DataFrame
        Columns ``process``, ``enrich_T``, ``enrich_F`` (and ``dominant``).
    metric : {"T_vs_F", "T"}
    eps : float
        Offset that keeps the log finite when a share is 0.
    ax : matplotlib Axes, optional
    title : str, optional
    output_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    apply_publication_style()

    required = ['process', 'enrich_T'] + (['enrich_F'] if metric == 'T_vs_F' else [])
    _require_columns(alignment_df, required, 'plot_process_alignment')
    if metric not in ('T_vs_F', 'T'):
        raise ValueError("metric must be 'T_vs_F' or 'T'")

    df = alignment_df.copy()
    enrich_T = df['enrich_T'].astype(float).to_numpy()
    if metric == 'T_vs_F':
        enrich_F = df['enrich_F'].astype(float).to_numpy()
        value = np.log2((enrich_T + eps) / (enrich_F + eps))
        xlabel = r'$\log_2$(T enrichment / F enrichment)'
        neg_label, pos_label = 'perturbation-aligned (F)', 'baseline-aligned (T)'
    else:
        value = np.log2(enrich_T + eps)
        xlabel = r'$\log_2$(T enrichment over chance)'
        neg_label, pos_label = 'less baseline than chance', 'more baseline than chance'
    df['_value'] = value
    df = df[np.isfinite(df['_value'])].sort_values('_value')

    names = []
    for _, row in df.iterrows():
        tag = ' (I)' if str(row.get('dominant', '')) == 'I' else ''
        names.append(f"{row['process']}{tag}")

    fig, ax, own = _get_axes(ax, (7.8, 0.32 * len(df) + 1.8))
    y = np.arange(len(df))
    vals = df['_value'].to_numpy()
    colors = np.where(vals >= 0, CHANNEL_COLORS['T'], CHANNEL_COLORS['F'])
    ax.barh(y, vals, height=0.62, color=colors, edgecolor=_SURFACE, linewidth=1.0)
    ax.axvline(0, color=_TEXT_SECONDARY, linewidth=1.0, zorder=3)
    lim = max(1.0, float(np.nanmax(np.abs(vals))) * 1.08) if len(vals) else 1.0
    ax.set_xlim(-lim, lim)
    ax.text(0.02 * lim, len(df) - 0.45, 'chance level', ha='left', va='bottom', fontsize=8,
            color=_TEXT_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(xlabel, color=_TEXT_SECONDARY)
    _recessive_axes(ax, grid_axis='x')

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CHANNEL_COLORS['T'], label=pos_label),
                       Patch(color=CHANNEL_COLORS['F'], label=neg_label)],
              loc='upper center', bbox_to_anchor=(0.5, 0.0), borderaxespad=4.0,
              ncol=2, frameon=False, fontsize=9)
    if title is None:
        title = 'Process alignment with the baseline vs perturbation subspaces'
    ax.set_title(title, fontsize=11, color=_TEXT_PRIMARY, loc='left')

    return _finish(fig, output_path, own)
