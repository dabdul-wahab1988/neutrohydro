"""
Tests for the minerals module.
"""

import numpy as np
import pytest

from neutrohydro.minerals import (
    MineralInverter,
    MineralInversionResult,
    build_stoichiometric_matrix,
    convert_to_meq,
    STANDARD_MINERALS,
    STANDARD_IONS,
    ION_MASSES,
    ION_CHARGES,
)


class TestBuildStoichiometricMatrix:
    """Tests for build_stoichiometric_matrix function."""

    def test_standard_minerals(self):
        """Test building matrix from standard minerals."""
        A, mineral_names, ion_names = build_stoichiometric_matrix(STANDARD_MINERALS)

        assert A.shape[0] == len(STANDARD_IONS)
        assert A.shape[1] == len(STANDARD_MINERALS)
        assert len(mineral_names) == len(STANDARD_MINERALS)
        assert ion_names == STANDARD_IONS

    def test_custom_minerals(self):
        """Test building matrix from custom minerals."""
        custom = {
            "Test1": {"formula": "NaCl", "stoichiometry": {"Na+": 1.0, "Cl-": 1.0}},
            "Test2": {"formula": "CaCO3", "stoichiometry": {"Ca2+": 2.0, "HCO3-": 2.0}},
        }

        A, mineral_names, ion_names = build_stoichiometric_matrix(custom)

        assert A.shape[1] == 2
        assert "Test1" in mineral_names
        assert "Test2" in mineral_names

    def test_stoichiometry_values(self):
        """Test that stoichiometry values are correct."""
        A, mineral_names, ion_names = build_stoichiometric_matrix(STANDARD_MINERALS)

        # Find halite (NaCl)
        halite_idx = mineral_names.index("Halite")
        na_idx = ion_names.index("Na+")
        cl_idx = ion_names.index("Cl-")

        assert A[na_idx, halite_idx] == 1.0
        assert A[cl_idx, halite_idx] == 1.0


class TestMineralInverter:
    """Tests for MineralInverter class."""

    def test_initialization(self):
        """Test basic initialization."""
        inverter = MineralInverter()

        assert inverter.n_ions == len(STANDARD_IONS)
        assert inverter.n_minerals == len(STANDARD_MINERALS)
        assert inverter.A.shape == (len(STANDARD_IONS), len(STANDARD_MINERALS))

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        inverter = MineralInverter(eta=2.0, tau_s=0.1, tau_r=2.0)

        assert inverter.eta == 2.0
        assert inverter.tau_s == 0.1
        assert inverter.tau_r == 2.0

    def test_invert_single_sample(self, meq_data):
        """Test inversion on single sample."""
        inverter = MineralInverter()
        c = meq_data[0, :]  # Single sample

        result = inverter.invert(c)

        assert isinstance(result, MineralInversionResult)
        assert result.s.shape == (1, inverter.n_minerals)
        assert len(result.residual_norms) == 1

    def test_invert_multiple_samples(self, meq_data):
        """Test inversion on multiple samples."""
        inverter = MineralInverter()
        result = inverter.invert(meq_data)

        n = meq_data.shape[0]
        assert result.s.shape == (n, inverter.n_minerals)
        assert result.residuals.shape == (n, inverter.n_ions)
        assert len(result.residual_norms) == n
        assert result.plausible.shape == (n, inverter.n_minerals)

    def test_s_non_negative(self, meq_data):
        """Test that mineral contributions are non-negative (NNLS)."""
        inverter = MineralInverter()
        result = inverter.invert(meq_data)

        assert np.all(result.s >= 0)

    def test_mineral_fractions_sum_to_one(self, meq_data):
        """Test that mineral fractions sum to approximately 1."""
        inverter = MineralInverter()
        result = inverter.invert(meq_data)

        row_sums = result.mineral_fractions.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=5)

    def test_weighted_inversion(self, meq_data):
        """Test inversion with pi_G weights."""
        inverter = MineralInverter()

        # Create dummy pi_G (baseline fractions)
        pi_G = np.random.uniform(0.3, 0.9, inverter.n_ions)

        result_unweighted = inverter.invert(meq_data)
        result_weighted = inverter.invert(meq_data, pi_G=pi_G)

        # Results should differ when weighted
        assert not np.allclose(result_unweighted.s, result_weighted.s)

    def test_plausibility_criteria(self, meq_data):
        """Test plausibility based on thresholds."""
        inverter = MineralInverter(tau_s=0.01, tau_r=10.0)
        result = inverter.invert(meq_data)

        # Plausible if s > tau_s and residual_norm <= tau_r
        for i in range(meq_data.shape[0]):
            for k in range(inverter.n_minerals):
                expected = (result.s[i, k] > inverter.tau_s and
                           result.residual_norms[i] <= inverter.tau_r)
                assert result.plausible[i, k] == expected

    def test_stoichiometry_dataframe(self):
        """Test get_stoichiometry_dataframe method."""
        inverter = MineralInverter()
        df = inverter.get_stoichiometry_dataframe()

        assert list(df.index) == inverter.ion_names
        assert list(df.columns) == inverter.mineral_names

    def test_results_to_dataframe(self, meq_data):
        """Test results_to_dataframe method."""
        inverter = MineralInverter()
        result = inverter.invert(meq_data)
        df = inverter.results_to_dataframe(result)

        n = meq_data.shape[0]
        assert len(df) == n
        assert 'sample_id' in df.columns
        assert 'residual_norm' in df.columns

    def test_wrong_ion_count_raises(self):
        """Test that wrong number of ions raises error."""
        inverter = MineralInverter()
        c = np.array([[1.0, 2.0, 3.0]])  # Wrong number of ions

        with pytest.raises(ValueError):
            inverter.invert(c)


class TestConvertToMeq:
    """Tests for convert_to_meq function."""

    def test_mg_to_meq(self):
        """Test conversion from mg/L to meq/L."""
        # 40.078 mg/L Ca2+ = 1 mmol/L = 2 meq/L
        concentrations = np.array([[40.078, 35.453]])  # Ca2+, Cl-
        ion_charges = {"Ca2+": 2, "Cl-": 1}
        ion_masses = {"Ca2+": 40.078, "Cl-": 35.453}

        result = convert_to_meq(concentrations, ion_charges, ion_masses, "mg/L")

        np.testing.assert_array_almost_equal(result[0], [2.0, 1.0], decimal=5)

    def test_mmol_to_meq(self):
        """Test conversion from mmol/L to meq/L."""
        concentrations = np.array([[1.0, 1.0]])  # 1 mmol/L each
        ion_charges = {"Ca2+": 2, "Cl-": 1}
        ion_masses = {"Ca2+": 40.078, "Cl-": 35.453}

        result = convert_to_meq(concentrations, ion_charges, ion_masses, "mmol/L")

        np.testing.assert_array_almost_equal(result[0], [2.0, 1.0])

    def test_invalid_unit_raises(self):
        """Test that invalid unit raises error."""
        concentrations = np.array([[1.0]])
        ion_charges = {"Ca2+": 2}
        ion_masses = {"Ca2+": 40.078}

        with pytest.raises(ValueError):
            convert_to_meq(concentrations, ion_charges, ion_masses, "invalid")
