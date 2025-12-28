"""
Integrated NeutroHydro Run: Quality Checks + Mineral Inversion.

This script demonstrates how WHO Quality Flags can 'help' the mineral inversion model
by providing context-aware constraints (e.g., forcing Evaporites in Saline Intrusion zones).
"""
import sys
import os
import pandas as pd
import numpy as np

# Ensure imports work - Insert at 0 to prioritize local source over installed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from neutrohydro.minerals import MineralInverter, convert_to_meq
from neutrohydro.quality_check import add_quality_flags

def main():
    # 1. Load Data
    data_path = "data3.csv"
    if not os.path.exists(data_path):
        print("Data file not found.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")

    # 2. Run Quality Assessment
    print("Running WHO Quality Assessment...")
    df_quality = add_quality_flags(df)
    
    # Extract flags as a list of dicts for the inverter
    quality_flags = df_quality.to_dict('records')
    
    # 3. Prepare Concentrations for Inversion
    # We need to map columns to standard ions
    # data3.csv cols: Na, K, Ca, Mg, HCO3, Cl, SO4, NO3, F
    ion_map = {
        "Na": "Na+", "K": "K+", "Ca": "Ca2+", "Mg": "Mg2+",
        "HCO3": "HCO3-", "Cl": "Cl-", "SO4": "SO42-",
        "NO3": "NO3-", "F": "F-"
    }
    
    # Create concentration matrix (mg/L)
    # We need to ensure we pass them in the order the Inverter expects, or use a dict
    inverter = MineralInverter()
    
    # Helper to build the matrix
    # We'll just build a simple matrix for the ions we have
    # The inverter expects specific columns if we pass a matrix, 
    # but we can use the helper `convert_to_meq` if we are careful.
    # Let's manually build the meq matrix to be safe and precise.
    
    c_meq = np.zeros((len(df), inverter.n_ions))
    
    # Standard masses/charges (simplified for this script, usually imported)
    # Better: use the inverter's internal data or the helper if available.
    # Let's use the helper `convert_to_meq` from minerals.py if we can import the dicts.
    from neutrohydro.minerals import ION_CHARGES, ION_MASSES
    
    for i, ion_name in enumerate(inverter.ion_names):
        # Find corresponding col in csv
        # Invert map: ion_name -> csv_col
        csv_col = None
        for k, v in ion_map.items():
            if v == ion_name:
                csv_col = k
                break
        
        if csv_col and csv_col in df.columns:
            # Convert mg/L to meq/L
            vals_mg = df[csv_col].values
            mass = ION_MASSES.get(ion_name, 1.0)
            charge = ION_CHARGES.get(ion_name, 1.0)
            c_meq[:, i] = (vals_mg / mass) * charge
            
    # 4. Run Inversion WITH Quality Flags
    print("Running Mineral Inversion with Quality Constraints...")
    result = inverter.invert(
        c=c_meq,
        use_cai_constraints=True,
        use_gibbs_constraints=True,
        quality_flags=quality_flags  # <--- The new integration
    )
    
    # 5. Export Results
    df_res = inverter.results_to_dataframe(result, sample_ids=df['Code'].tolist())
    
    # Merge with Quality Info for final report
    final_df = pd.concat([
        df_quality[['Code', 'Exceedances', 'Inferred_Sources']], 
        df_res.drop(columns=['sample_id'])
    ], axis=1)
    
    output_path = "integrated_results.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\nIntegrated analysis saved to {output_path}")
    print("\nSample Output (First 3 rows):")
    print(final_df[['Code', 'Inferred_Sources', 'Simpson_Class', 'Halite_frac']].head(3))

if __name__ == "__main__":
    main()
