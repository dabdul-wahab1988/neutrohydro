"""
Example: Generate hydrogeochemical visualization report.

This script demonstrates how to use NeutroHydro's visualization module
to create publication-quality plots from groundwater chemistry data.

Shows three approaches:
1. generate_report() - Complete automated report
2. Individual plotting functions - Fine-grained control
3. Registry-based composition - Modular plot assembly

Usage:
    python examples/run_visualization.py
"""

import pandas as pd
import numpy as np
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neutrohydro.visualization import (
    generate_report,
    plot_gibbs,
    plot_ilr_classification,
    plot_correlation_matrix,
    create_figure,
    mg_to_meq,
)


def main():
    # 1. Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data3.csv')
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")
    
    # 2. Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Option A: Generate full report (easiest for non-coders)
    print("\n" + "="*60)
    print("Generating Full Report")
    print("="*60)
    generate_report(df, output_dir=output_dir, preset='core')
    
    # 4. Option B: Generate individual plots with more control
    print("\n" + "="*60)
    print("Generating Individual Plots")
    print("="*60)
    
    # Convert to meq/L
    df_meq = mg_to_meq(df)
    
    # Gibbs plot with custom output
    fig = plot_gibbs(df, df_meq, output_path=os.path.join(output_dir, 'gibbs_custom.png'))
    print("  [OK] gibbs_custom.png")
    
    # ILR classification
    fig = plot_ilr_classification(df, df_meq, output_path=os.path.join(output_dir, 'ilr_custom.png'))
    print("  [OK] ilr_custom.png")
    
    # Correlation matrix with specific columns
    ion_cols = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4', 'NO3', 'F']
    available_cols = [c for c in ion_cols if c in df.columns]
    fig = plot_correlation_matrix(df, columns=available_cols, 
                                   output_path=os.path.join(output_dir, 'correlation_ions.png'))
    print("  [OK] correlation_ions.png")
    
    # 5. Option C: Registry-based composition (advanced)
    print("\n" + "="*60)
    print("Registry-Based Plot Composition")
    print("="*60)
    
    # Prepare data dictionary for registry
    registry_data = {'df': df, 'df_meq': df_meq}
    
    # Dual Gibbs plot using registry
    fig = create_figure('[g1|g2]', registry_data, 
                       output_path=os.path.join(output_dir, 'registry_gibbs_dual.png'))
    print("  [OK] registry_gibbs_dual.png")
    
    # Correlation matrix using registry
    fig = create_figure('[h1]', registry_data,
                       output_path=os.path.join(output_dir, 'registry_correlation.png'))
    print("  [OK] registry_correlation.png")
    
    # Use preset configurations
    fig = create_figure('fig1_classification', registry_data,
                       output_path=os.path.join(output_dir, 'preset_classification.png'))
    print("  [OK] preset_classification.png")
    
    fig = create_figure('fig2_correlation', registry_data,
                       output_path=os.path.join(output_dir, 'preset_correlation.png'))
    print("  [OK] preset_correlation.png")
    
    print("\n" + "="*60)
    print(f"\n[OK] All plots saved to: {os.path.abspath(output_dir)}")
    print("Individual functions: gibbs_custom.png, ilr_custom.png, correlation_ions.png")
    print("Registry composition: registry_gibbs_dual.png, registry_correlation.png")
    print("Preset layouts: preset_classification.png, preset_correlation.png")
    print("="*60)


if __name__ == '__main__':
    main()
