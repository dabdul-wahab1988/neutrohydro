import numpy as np
import pandas as pd

# WHO Guideline Limits (mg/L)
WHO_LIMITS = {
    "TDS": 1000.0,
    "pH_min": 6.5,
    "pH_max": 8.5,
    "Na": 200.0,
    "K": 64,
    "Ca": 200.0,
    "Mg": 100.0,
    "Cl": 250.0,
    "SO4": 250.0,
    "NO3": 50.0,
    "F": 1.5,
    "Fe": 0.3,
    "Mn": 0.1,
    "Zn": 3.0,
    "Pb": 0.01,
    "As": 0.01,
    "Cd": 0.003,
    "Cr": 0.05,
    "Cu": 2.0,
}


def assess_water_quality(row: dict) -> dict:
    """
    Assess water quality against WHO guidelines and infer sources.

    Parameters
    ----------
    row : dict
        Dictionary of parameter values (mg/L). Keys should match standard chemical symbols (e.g., 'Na', 'Cl', 'NO3').

    Returns
    -------
    dict
        Dictionary containing 'Exceedances' (list), 'Pollution_Index' (int), and 'Inferred_Source' (str).
    """
    exceedances = []
    sources = set()

    # Check Limits
    if row.get("TDS", 0) > WHO_LIMITS["TDS"]:
        exceedances.append("TDS")

    ph = row.get("pH")
    if ph is not None:
        if ph < WHO_LIMITS["pH_min"]:
            exceedances.append("pH (Acidic)")
            sources.add("Industrial/Acid Rain")
        elif ph > WHO_LIMITS["pH_max"]:
            exceedances.append("pH (Alkaline)")

    for ion, limit in WHO_LIMITS.items():
        if ion in ["TDS", "pH_min", "pH_max"]:
            continue
        val = row.get(ion)
        if val is not None and val > limit:
            exceedances.append(ion)

            # Source Inference Logic
            if ion == "NO3":
                sources.add("Anthropogenic (Agri/Sewage)")
            elif ion == "F":
                sources.add("Geogenic (Rock-Water)")
            elif ion == "Cl":
                if row.get("Na", 0) > WHO_LIMITS["Na"]:
                    sources.add("Saline Intrusion/Brine")
                else:
                    sources.add("Anthropogenic/Industrial")
            elif ion == "SO4":
                if row.get("Ca", 0) > WHO_LIMITS["Ca"]:
                    sources.add("Gypsum/Evaporites")
                else:
                    sources.add("Industrial/Mining")
            elif ion in ["Pb", "Cd", "Cr", "As"]:
                sources.add("Industrial/Toxic Waste")

    return {
        "Exceedances": ", ".join(exceedances) if exceedances else "None",
        "Pollution_Count": len(exceedances),
        "Inferred_Sources": ", ".join(sorted(list(sources))) if sources else "Natural/Safe",
    }


def add_quality_flags(df):
    """
    Add WHO quality assessment columns to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with chemical data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added 'Exceedances', 'Pollution_Count', and 'Inferred_Sources' columns.
    """
    import pandas as pd

    results = []
    for _, row in df.iterrows():
        # Convert row to dict, handling potential NaN
        row_dict = row.to_dict()
        results.append(assess_water_quality(row_dict))

    quality_df = pd.DataFrame(results)
    # Concatenate while preserving index
    return pd.concat([df.reset_index(drop=True), quality_df], axis=1)


REQUIRED_IONS = ["Ca", "Mg", "Na", "K", "HCO3", "Cl", "SO4"]

def check_data_completeness(df_columns: list[str]) -> list[str]:
    """
    Check if all 7 major ions are present in the dataset.
    
    Parameters
    ----------
    df_columns : list[str]
        List of column names in the dataframe.
        
    Returns
    -------
    list[str]
        List of missing ions.
    """
    missing = []
    for ion in REQUIRED_IONS:
        found = False
        for col in df_columns:
            # Check for "Ca", "Ca2+", "Calcium", etc.
            if ion.lower() == col.lower() or \
               f"{ion}+".lower() in col.lower() or \
               f"{ion}-".lower() in col.lower() or \
               f"{ion}2+".lower() in col.lower() or \
               f"{ion}2-".lower() in col.lower():
                found = True
                break
        if not found:
            missing.append(ion)
            
    return missing


def calculate_cbe(row: dict) -> float:
    """
    Calculate Charge Balance Error (CBE).
    
    CBE = (Sum Cations - Sum Anions) / (Sum Cations + Sum Anions) * 100
    
    Assumes units are mg/L and converts to meq/L.
    """
    # Molar masses (mg/mmol) and valences
    factors = {
        "Ca": 2 / 40.08,
        "Mg": 2 / 24.305,
        "Na": 1 / 22.99,
        "K": 1 / 39.098,
        "HCO3": 1 / 61.017,
        "Cl": 1 / 35.45,
        "SO4": 2 / 96.06,
        "NO3": 1 / 62.005,
        "CO3": 2 / 60.01,
    }
    
    cations = 0.0
    anions = 0.0
    
    # Helper to find value with flexible keys
    def get_val(ion):
        # Try exact match
        if ion in row: return row[ion]
        # Try with charge
        for k in row.keys():
            if k.lower().startswith(ion.lower()):
                return row[k]
        return 0.0

    cations += get_val("Ca") * factors["Ca"]
    cations += get_val("Mg") * factors["Mg"]
    cations += get_val("Na") * factors["Na"]
    cations += get_val("K") * factors["K"]
    
    anions += get_val("HCO3") * factors["HCO3"]
    anions += get_val("Cl") * factors["Cl"]
    anions += get_val("SO4") * factors["SO4"]
    anions += get_val("NO3") * factors["NO3"]
    anions += get_val("CO3") * factors["CO3"]
    
    if cations + anions == 0:
        return 0.0
        
    return (cations - anions) / (cations + anions) * 100.0


def check_sanity(df: pd.DataFrame) -> dict:
    """
    Run all sanity checks on the dataframe.
    """
    report = {
        "missing_ions": [],
        "high_cbe_count": 0,
        "extreme_samples": 0,
        "valid": True,
        "warnings": []
    }
    
    # 1. Completeness
    report["missing_ions"] = check_data_completeness(df.columns.tolist())
    if report["missing_ions"]:
        report["valid"] = False
        report["warnings"].append(f"Missing critical ions: {report['missing_ions']}")
        
    # 2. Balance
    cbes = []
    for _, row in df.iterrows():
        cbes.append(abs(calculate_cbe(row.to_dict())))
    
    high_cbe = [c for c in cbes if c > 15.0]
    report["high_cbe_count"] = len(high_cbe)
    if len(high_cbe) > 0.2 * len(df):
        report["warnings"].append(f"High Charge Balance Error (>15%) in {len(high_cbe)} samples.")
        
    # 3. Extreme Contamination (Swamp Effect)
    extreme_count = 0
    for _, row in df.iterrows():
        # Check Cl > 10,000 mg/L
        cl = 0
        for k in row.keys():
            if k.lower().startswith("cl"):
                cl = row[k]
                break
        if cl > 10000:
            extreme_count += 1
            
    report["extreme_samples"] = extreme_count
    if extreme_count > 0:
        report["warnings"].append(f"Found {extreme_count} samples with extreme contamination (Cl > 10,000 mg/L).")

    return report
