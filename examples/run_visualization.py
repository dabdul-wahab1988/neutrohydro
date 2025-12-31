"""
Example: Generate hydrogeochemical visualization report.

This script demonstrates how to use NeutroHydro's visualization module
to create publication-quality plots from groundwater chemistry data.

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
    
    print("\n" + "="*60)
    print(f"\n[OK] All plots saved to: {os.path.abspath(output_dir)}")
    print("="*60)


if __name__ == '__main__':
    main()
