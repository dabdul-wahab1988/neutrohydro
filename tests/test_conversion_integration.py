"""Integration tests for unit conversions used in examples."""
import numpy as np
import pandas as pd

from examples.run_data3 import prepare_c_meq
from neutrohydro.minerals import ION_MASSES, ION_CHARGES


def test_prepare_c_meq_first_row_matches_manual():
    df = pd.read_csv("data3.csv")

    c_meq, cols_ordered = prepare_c_meq(df, ['Na','K','Ca','Mg','HCO3','Cl','SO4','NO3','F'])

    # Expected ordered short names corresponding to inverter ion order
    expected_order = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4', 'NO3', 'F']
    assert cols_ordered == expected_order

    # Manual calculation for first sample and first ion (Ca)
    ca_mgL = df.loc[0, 'Ca']
    ca_mass = ION_MASSES['Ca2+']
    ca_charge = ION_CHARGES['Ca2+']

    expected_ca_meq = (ca_mgL / ca_mass) * abs(ca_charge)

    assert np.isclose(c_meq[0, 0], expected_ca_meq, atol=1e-9)


def test_prepare_c_meq_all_match_convert_to_meq():
    # Sanity check: manual per-column conversion equals vectorized conversion formula
    df = pd.read_csv("data3.csv")
    c_meq, cols_ordered = prepare_c_meq(df, ['Na','K','Ca','Mg','HCO3','Cl','SO4','NO3','F'])

    # Recompute per column using ION_MASSES and ION_CHARGES
    expected = np.zeros_like(c_meq)
    for j, short in enumerate(cols_ordered):
        std_name = {
            'Na': 'Na+', 'K': 'K+', 'Ca': 'Ca2+', 'Mg': 'Mg2+', 'HCO3': 'HCO3-',
            'Cl': 'Cl-', 'SO4': 'SO42-', 'NO3': 'NO3-', 'F': 'F-'
        }[short]
        mass = ION_MASSES[std_name]
        charge = ION_CHARGES[std_name]
        expected[:, j] = (df[short].values.astype(float) / mass) * abs(charge)

    assert np.allclose(c_meq, expected, atol=1e-9)
