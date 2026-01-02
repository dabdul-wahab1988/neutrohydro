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
