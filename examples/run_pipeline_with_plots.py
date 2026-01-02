"""
Example: Complete NeutroHydro Pipeline with Model Visualization.

This script demonstrates the full NeutroHydro workflow:
1. Load groundwater chemistry data
2. Run the NeutroHydro pipeline (encoding + modeling + NVIP)
3. Generate comprehensive visualizations including model diagnostics

Shows both individual plotting functions and registry-based composition.

Usage:
    python examples/run_pipeline_with_plots.py
"""

import pandas as pd
import numpy as np
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neutrohydro.pipeline import NeutroHydroPipeline, PipelineConfig
from neutrohydro.visualization import (
    generate_report,
    create_figure,
    mg_to_meq,
)


def main():
    # 1. Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data3.csv')
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

    # 2. Prepare data for modeling
    # Use major ions as predictors, TDS as target
    X = df[['Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4', 'NO3']].values
    y = df['TDS'].values
    feature_names = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4', 'NO3']

    # 3. Configure and run NeutroHydro pipeline
    print("\n" + "="*60)
    print("Running NeutroHydro Pipeline")
    print("="*60)

    config = PipelineConfig(
        baseline_type='robust_pca',  # Default robust PCA
        baseline_rank=2,             # 2D geological baseline
        n_components=3,              # 3 PLS components
        run_mineral_inference=False, # Skip minerals for this example
    )

    pipeline = NeutroHydroPipeline(config)
    results = pipeline.fit(X, y, feature_names=feature_names)

    print(".3f")
    print(f"NVIP analysis: {results.nvip is not None}")

    # 4. Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'pipeline_figures')
    os.makedirs(output_dir, exist_ok=True)

    # 5. Generate basic hydrochemical plots
    print("\n" + "="*60)
    print("Generating Hydrochemical Plots")
    print("="*60)

    generate_report(df, output_dir=output_dir, preset='core')
    print("  [OK] Basic report generated")

    # 6. Generate model-specific plots using registry
    print("\n" + "="*60)
    print("Generating Model Diagnostic Plots")
    print("="*60)

    # Prepare data for registry plotting
    df_meq = mg_to_meq(df)
    plot_data = {
        'df': df,
        'df_meq': df_meq,
        'nvip_result': results.nvip
    }

    # PLS loadings by channel
    fig = create_figure('[p1]', plot_data,
                       output_path=os.path.join(output_dir, 'pls_loadings.png'))
    print("  [OK] pls_loadings.png")

    # Explained variance scree plot
    fig = create_figure('[p2]', plot_data,
                       output_path=os.path.join(output_dir, 'explained_variance.png'))
    print("  [OK] explained_variance.png")

    # Combined model diagnostics
    fig = create_figure('[p1][p2]', plot_data,
                       output_path=os.path.join(output_dir, 'model_diagnostics.png'),
                       figsize=(10, 8))
    print("  [OK] model_diagnostics.png")

    # 7. Generate VIP plots (if available)
    if results.nvip is not None:
        print("\n" + "="*60)
        print("Generating VIP Analysis Plots")
        print("="*60)

        # VIP decomposition by channel
        fig = create_figure('[v1]', plot_data,
                           output_path=os.path.join(output_dir, 'vip_decomposition.png'))
        print("  [OK] vip_decomposition.png")

        # VIP aggregate
        fig = create_figure('[v2]', plot_data,
                           output_path=os.path.join(output_dir, 'vip_aggregate.png'))
        print("  [OK] vip_aggregate.png")

        # Baseline fraction (πG)
        fig = create_figure('[v3]', plot_data,
                           output_path=os.path.join(output_dir, 'baseline_fraction.png'))
        print("  [OK] baseline_fraction.png")

        # Sample G distribution
        fig = create_figure('[v4]', plot_data,
                           output_path=os.path.join(output_dir, 'sample_g_distribution.png'))
        print("  [OK] sample_g_distribution.png")

    # 8. Create comprehensive dashboard
    print("\n" + "="*60)
    print("Creating Comprehensive Dashboard")
    print("="*60)

    # Multi-panel figure with key results
    dashboard_data = {
        'df': df,
        'df_meq': df_meq,
        'nvip_result': results.nvip
    }

    # Layout: Gibbs + Correlation on top, Model diagnostics below
    fig = create_figure('[g1|g2][h1][p1][p2]', dashboard_data,
                       output_path=os.path.join(output_dir, 'comprehensive_dashboard.png'),
                       figsize=(14, 12))
    print("  [OK] comprehensive_dashboard.png")

    print("\n" + "="*60)
    print(f"[SUCCESS] Complete analysis saved to: {os.path.abspath(output_dir)}")
    print("Hydrochemical: gibbs_plot.png, ilr_classification.png, correlation_matrix.png")
    print("Model diagnostics: pls_loadings.png, explained_variance.png, model_diagnostics.png")
    if results.nvip is not None:
        print("VIP analysis: vip_decomposition.png, vip_aggregate.png, baseline_fraction.png")
    print("Dashboard: comprehensive_dashboard.png")
    print("="*60)


if __name__ == '__main__':
    main()