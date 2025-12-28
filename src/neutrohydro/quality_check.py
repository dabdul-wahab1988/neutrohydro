
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
