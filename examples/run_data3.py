"""
Run NeutroHydro pipeline on the provided `data3.csv` groundwater dataset.

This script performs two runs to compare baseline strategies and includes
mineral apportionments (converted to meq/L) using the built-in inverter.

Usage:
    C:/.../.venv/Scripts/python.exe examples/run_data3.py data3.csv

Assumes the package (from source) is importable from the workspace environment.
"""
import sys
import os
import numpy as np
import pandas as pd

from neutrohydro import NeutroHydroPipeline
from neutrohydro.pipeline import PipelineConfig
from neutrohydro.minerals import (
    MineralInverter,
    convert_to_meq,
    ION_CHARGES,
    ION_MASSES,
    STANDARD_MINERALS,
    EXCHANGER_PHASES,
)


def prepare_c_meq(df, input_cols):
    """
    Prepare concentrations in meq/L ordered to the MineralInverter standard ion order.
    Only includes ions present in the input dataframe.
    """
    inverter = MineralInverter()
    # Mapping from short column names to standard ion names
    short_to_standard = {
        'Na': 'Na+',
        'K': 'K+',
        'Ca': 'Ca2+',
        'Mg': 'Mg2+',
        'HCO3': 'HCO3-',
        'Cl': 'Cl-',
        'SO4': 'SO42-',
        'NO3': 'NO3-',
        'F': 'F-',
        'Zn': 'Zn2+',
        'Cd': 'Cd2+',
        'Pb': 'Pb2+',
        'B': 'B',
        'Cu': 'Cu2+',
        'As': 'As',
        'Cr': 'Cr',
        'U': 'U',
    }

    # Build matrix in inverter's ion order
    c_mgL_list = []
    found_ions = []
    
    for ion in inverter.ion_names:
        # find which short name maps to this ion
        found_col = None
        for short, std in short_to_standard.items():
            if std == ion:
                found_col = short
                break
        
        if found_col and found_col in input_cols:
            # Column exists, use it
            c_mgL_list.append(df[found_col].values.astype(float))
            found_ions.append(ion)

    if not c_mgL_list:
        return None, []

    # Stack to (n_samples, n_ions)
    c_mgL = np.column_stack(c_mgL_list)
    
    # Filter charges and masses to match found_ions order
    filtered_charges = {ion: ION_CHARGES[ion] for ion in found_ions}
    filtered_masses = {ion: ION_MASSES[ion] for ion in found_ions}
    
    # Convert to meq/L
    c_meq = convert_to_meq(c_mgL, filtered_charges, filtered_masses, from_unit='mg/L')

    return c_meq, found_ions


def run_pipeline_with_config(df, feature_cols, config: PipelineConfig, outdir: str, c_meq_data=None):
    """
    c_meq_data: tuple (c_meq_array, found_ion_names)
    """
    os.makedirs(outdir, exist_ok=True)

    X = df[feature_cols].values.astype(float)
    y = np.log(np.maximum(df['TDS'].values.astype(float), 1.0))

    # Disable internal mineral inference to handle ion mismatch manually
    config.run_mineral_inference = False
    
    pipeline = NeutroHydroPipeline(config)
    results = pipeline.fit(X, y, feature_names=feature_cols)

    # Save standard outputs
    dfs = pipeline.to_dataframes()
    for name, dfout in dfs.items():
        dfout.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)

    # Save summary
    summary = pipeline.get_summary()
    import json
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Manual Mineral Inference with filtered ion list
    if c_meq_data is not None:
        c_meq, found_ions = c_meq_data
        
        # Filter minerals: Only keep minerals whose stoichiometry keys are ALL in found_ions
        active_minerals = {}
        # Combine Standard and Exchanger phases
        all_candidates = {**STANDARD_MINERALS, **EXCHANGER_PHASES}
        
        for m_name, m_def in all_candidates.items():
            required_ions = m_def['stoichiometry'].keys()
            if all(ion in found_ions for ion in required_ions):
                active_minerals[m_name] = m_def
        
        # Initialize inverter with filtered lists
        inverter = MineralInverter(
            minerals=active_minerals,
            ion_order=found_ions,
            eta=config.mineral_eta,
            tau_s=config.mineral_tau_s,
            tau_r=config.mineral_tau_r,
        )
        
        # Construct pi_G vector matching found_ions
        pi_G_filtered = np.zeros(len(found_ions))
        
        # Map feature_cols (short names) to standard names
        short_to_standard = {
            'Na': 'Na+', 'K': 'K+', 'Ca': 'Ca2+', 'Mg': 'Mg2+',
            'HCO3': 'HCO3-', 'Cl': 'Cl-', 'SO4': 'SO42-', 'NO3': 'NO3-', 'F': 'F-'
        }
        
        # results.nsr.pi_G corresponds to feature_cols order
        for i, feat in enumerate(feature_cols):
            std_name = short_to_standard.get(feat)
            if std_name and std_name in found_ions:
                idx = found_ions.index(std_name)
                pi_G_filtered[idx] = results.nsr.pi_G[i]
        
        # Run inversion with advanced constraints (CAI + Gibbs)
        mineral_result = inverter.invert(
            c_meq, 
            pi_G_filtered,
            use_cai_constraints=True,
            use_gibbs_constraints=True
        )
        
        mineral_df = inverter.results_to_dataframe(mineral_result)
        mineral_df.to_csv(os.path.join(outdir, 'mineral_inversion.csv'), index=False)

        # Print short mineral summary
        plausible_counts = mineral_df.filter(like='_plausible').sum()
        print(f"  Mineral plausible counts:\n{plausible_counts.to_string()}\n")

    # Print concise summary to console
    print(f"Run: {outdir}")
    print(f"  Samples: {X.shape[0]}, Ions: {X.shape[1]}")
    print(f"  Model R² (train): {results.r2_train:.4f}")
    print("  NVIP (top 5):")
    print(dfs['nvip'].sort_values('VIP_agg', ascending=False).head().to_string(index=False))
    print("\n  NSR classification counts:")
    print(dfs['nsr']['classification'].value_counts().to_string())

    G = results.sample_attribution.G
    print(f"\n  Sample G stats: mean={np.mean(G):.3f}, std={np.std(G):.3f}, min={np.min(G):.3f}, max={np.max(G):.3f}\n")

    return results


def main(path='data3.csv'):
    df = pd.read_csv(path)

    # Input short ion columns present in data3.csv
    feature_cols = ['Na', 'K', 'Ca', 'Mg', 'HCO3', 'Cl', 'SO4', 'NO3', 'F']
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required ion columns: {missing}")
    if 'TDS' not in df.columns:
        raise RuntimeError('TDS column not found in CSV; cannot compute target')

    # Prepare c_meq for mineral inversion (aligned to inverter ion order)
    c_meq_data = prepare_c_meq(df, feature_cols)

    # Run 1: Low-rank baseline (rank=1) + mineral inference
    cfg1 = PipelineConfig(
        n_components=5,
        log_transform=False,
        baseline_type='low_rank',
        baseline_rank=1,
        rho_I=1.0,
        rho_F=1.0,
        lambda_F=1.0,
        gamma=0.7,
        run_mineral_inference=True,
    )
    run_pipeline_with_config(df, feature_cols, cfg1, outdir='results_lowrank', c_meq_data=c_meq_data)

    # Run 2: Robust PCA baseline (rank=2) + mineral inference
    cfg2 = PipelineConfig(
        n_components=5,
        log_transform=False,
        baseline_type='robust_pca',
        baseline_rank=2,
        rho_I=1.0,
        rho_F=1.0,
        lambda_F=1.0,
        gamma=0.7,
        run_mineral_inference=True,
    )
    run_pipeline_with_config(df, feature_cols, cfg2, outdir='results_robustpca', c_meq_data=c_meq_data)

    print('All runs complete. Results saved to results_lowrank/ and results_robustpca/')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data3.csv'
    main(path)

