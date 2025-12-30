"""
Run example with PHREEQC thermodynamic validation using data3.csv.
"""
import pandas as pd
import numpy as np
import os
from neutrohydro.pipeline import NeutroHydroPipeline, PipelineConfig
from neutrohydro.minerals import ION_MASSES, ION_CHARGES, convert_to_meq

def main():
    # 1. Load data
    df = pd.read_csv("data3.csv")
    
    # 2. Setup features and target
    feature_cols = ['Na', 'K', 'Ca', 'Mg', 'HCO3', 'Cl', 'SO4', 'NO3', 'F']
    X = df[feature_cols].values
    y = np.log(df['TDS'].values)
    pH = df['pH'].values
    Eh = df['Eh'].values
    
    # 3. Prepare meq/L for mineral inference
    # Mapping to standard ions
    short_to_std = {
        'Na': 'Na+', 'K': 'K+', 'Ca': 'Ca2+', 'Mg': 'Mg2+',
        'HCO3': 'HCO3-', 'Cl': 'Cl-', 'SO4': 'SO42-', 'NO3': 'NO3-', 'F': 'F-'
    }
    
    # Standard order for inverter
    standard_ion_order = [
        "Ca2+", "Mg2+", "Na+", "K+", "HCO3-", "Cl-", "SO42-", "NO3-", "F-",
        "Zn2+", "Cd2+", "Pb2+", "B", "Cu2+", "As", "Cr", "U"
    ]
    
    # Build meq/L matrix aligned to standard order
    c_meq = np.zeros((len(df), len(standard_ion_order)))
    for i, std_ion in enumerate(standard_ion_order):
        # find matching short name
        short = None
        for s, full in short_to_std.items():
            if full == std_ion:
                short = s
                break
        
        if short and short in df.columns:
            mass = ION_MASSES[std_ion]
            charge = ION_CHARGES[std_ion]
            c_meq[:, i] = (df[short].values / mass) * abs(charge)
            
    # 4. Configure Pipeline with Thermodynamics
    config = PipelineConfig(
        run_mineral_inference=True,
        run_thermodynamic_validation=True, # <--- ENABLED
        si_threshold=0.5,
        n_components=3
    )
    
    # 5. Run Pipeline
    print("Running NeutroHydro with PHREEQC Thermodynamic Validation...")
    pipeline = NeutroHydroPipeline(config)
    results = pipeline.fit(X, y, feature_names=feature_cols, c_meq=c_meq, pH=pH, Eh=Eh)
    
    # 6. Inspect Mineral Results
    if results.mineral_result:
        mr = results.mineral_result
        print("\n--- Mineral Inversion Results (Sample 0) ---")
        
        # Fractions
        fractions = mr.mineral_fractions[0]
        active_idx = np.where(fractions > 0.01)[0]
        
        for idx in active_idx:
            name = mr.mineral_names[idx]
            
            si = mr.saturation_indices[name][0] if mr.saturation_indices and name in mr.saturation_indices else "N/A"
            plausible = mr.plausible[0, idx]
            thermo_plausible = mr.thermo_plausible[0, idx] if mr.thermo_plausible is not None else "N/A"
            
            print(f"Mineral: {name:15} | Fraction: {fractions[idx]:.3f} | SI: {si:>6} | Plausible: {plausible} | Thermo-Plausible: {thermo_plausible}")

    print("\nProcessing complete. Saturation indices calculated via PHREEQC.")

if __name__ == "__main__":
    main()
