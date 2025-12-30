"""
Thermodynamic speciation and saturation index calculation using PHREEQC.
"""

import subprocess
import sys
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PHREEQC SETUP ---
PHREEQC_BACKEND = None

def install_phreeqpython():
    """Auto-install phreeqpython for scientific accuracy."""
    try:
        import phreeqpython
        return True
    except ImportError:
        logger.info("Installing phreeqpython for thermodynamic validation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "phreeqpython"])
            import phreeqpython
            return True
        except Exception as e:
            logger.error(f"Failed to install phreeqpython: {e}")
            return False

# Initialize backend
if install_phreeqpython():
    import phreeqpython
    PHREEQC_BACKEND = 'phreeqpython'
else:
    raise RuntimeError(
        "PHREEQC backend (phreeqpython) is required for thermodynamic validation. "
        "Auto-installation failed. Please install it manually: pip install phreeqpython"
    )

@dataclass
class ThermodynamicResult:
    """Container for saturation indices and feasibility flags."""
    saturation_indices: Dict[str, NDArray[np.floating]]
    feasible: NDArray[np.bool_]  # True if dissolution is possible (SI < threshold)

class ThermodynamicValidator:
    """
    Validates mineral dissolution feasibility using PHREEQC saturation indices.
    """

    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize with a specific PHREEQC database.
        
        Parameters
        ----------
        database_path : str, optional
            Path to the PHREEQC database file (.dat).
            Defaults to bundled wateq4f.dat if not provided.
        """
        if database_path is None:
            # Try to use bundled wateq4f.dat by default for better trace metal support
            base_dir = os.path.dirname(__file__)
            db_dir = os.path.join(base_dir, "databases")
            database_path = os.path.join(db_dir, "wateq4f.dat")
            
            if not os.path.exists(database_path):
                # Fallback to phreeqc.dat if wateq4f is missing
                database_path = os.path.join(db_dir, "phreeqc.dat")
            
            if not os.path.exists(database_path):
                # Fallback to defaults provided by phreeqpython
                database_path = None
                logger.warning("Bundled databases not found. Using phreeqpython default.")
        
        self.pp = phreeqpython.PhreeqPython(database=database_path)
        # Map common names to PHREEQC mineral names
        self.mineral_map = {
            "Calcite": "Calcite",
            "Dolomite": ["Dolomite", "Dolomite(d)"],
            "Magnesite": "Magnesite",
            "Gypsum": "Gypsum",
            "Anhydrite": "Anhydrite",
            "Halite": "Halite",
            "Sylvite": "Sylvite",
            "Mirabilite": "Mirabilite",
            "Thenardite": "Thenardite",
            "Fluorite": "Fluorite",
            "Albite": "Albite",
            "Anorthite": "Anorthite",
            "Kfeldspar": ["K-feldspar", "Adularia"],
            "Biotite": ["Biotite", "Phlogopite"],
            "Niter": "Niter",
            "SodaNiter": "Soda_Niter",
            "Nitrocalcite": "Nitrocalcite",
            "Otavite": "Otavite",
            "Smithsonite": "Smithsonite",
            "Cerussite": "Cerussite",
            "Borax": "Borax",
            "Malachite": "Malachite",
        }
        
        # Add missing common phases to the in-memory database if they are likely missing
        # This helps when using specialized databases like wateq4f which omit some basics
        self._add_fallback_phases()

    def _add_fallback_phases(self):
        """Add common mineral definitions if missing from the current database."""
        # Sylvite: KCl = K+ + Cl-; log_k 0.9
        # Niter: KNO3 = K+ + NO3-; log_k 0.6 approx
        # Soda_Niter: NaNO3 = Na+ + NO3-; log_k 0.0 approx
        
        fallback_script = """
PHASES
Sylvite
    KCl = K+ + Cl-
    log_k 0.9
Niter
    KNO3 = K+ + NO3-
    log_k 0.6
Soda_Niter
    NaNO3 = Na+ + NO3-
    log_k 0.0
"""
        # We try to add them using the raw interpreter
        try:
            self.pp.ip.run_string(fallback_script)
        except Exception as e:
            logger.warning(f"Failed to add fallback phases: {e}")

    def calculate_si(
        self,
        c: NDArray[np.floating],
        ion_names: List[str],
        pH: NDArray[np.floating],
        temp: float = 25.0,
        pe: Optional[NDArray[np.floating]] = None,
        Eh: Optional[NDArray[np.floating]] = None
    ) -> Dict[str, NDArray[np.floating]]:
        """
        Calculate Saturation Indices for all known minerals.
        
        Parameters
        ----------
        c : ndarray of shape (n_samples, n_ions)
            Concentrations in meq/L.
        ion_names : list of str
            Names matching columns of c.
        pH : ndarray of shape (n_samples,)
            pH values.
        temp : float
            Temperature in Celsius.
        pe : ndarray, optional
            Redox potential (pe).
        Eh : ndarray, optional
            Redox potential in mV. Used to calculate pe if pe is None.
            pe = Eh / (59.16 * (T[K]/298.15))
            
        Returns
        -------
        si_results : dict
            Dictionary mapping mineral names to SI arrays.
        """
        n_samples = c.shape[0]
        si_results = {name: np.full(n_samples, -999.0) for name in self.mineral_map.keys()}

        # Handle pe/Eh
        if pe is None:
            if Eh is not None:
                # pe = Eh(mV) / 59.16 at 25C
                pe_vals = Eh / (59.16 * (temp + 273.15) / 298.15)
            else:
                pe_vals = np.full(n_samples, 4.0) # Default oxidizing
        else:
            pe_vals = pe

        # Map standard ion names to PHREEQC component names
        # This assumes ion_names follow the STANDARD_IONS convention
        phreeqc_map = {
            "Ca2+": "Ca",
            "Mg2+": "Mg",
            "Na+": "Na",
            "K+": "K",
            "Cl-": "Cl",
            "SO42-": "S(6)",
            "HCO3-": "C(4)",
            "NO3-": "N(5)",
            "F-": "F",
            "Zn2+": "Zn",
            "Cd2+": "Cd",
            "Pb2+": "Pb",
            "B": "B",
            "Cu2+": "Cu",
            "As": "As",
            "Cr": "Cr",
            "U": "U(6)"
        }

        for i in range(n_samples):
            # Create solution
            sol_data = {
                'temp': temp,
                'pH': pH[i],
                'pe': pe_vals[i],
                'units': 'meq/l'
            }
            
            # Add ions
            for j, ion in enumerate(ion_names):
                p_name = phreeqc_map.get(ion)
                if p_name:
                    sol_data[p_name] = c[i, j]

            try:
                sol = self.pp.add_solution(sol_data)
                
                # Get SI for each mineral
                for common_name, phreeqc_names in self.mineral_map.items():
                    if isinstance(phreeqc_names, str):
                        phreeqc_names = [phreeqc_names]
                    
                    for p_name in phreeqc_names:
                        try:
                            val = sol.si(p_name)
                            # If we get a valid value (not -999), we stop
                            if val != -999.0:
                                si_results[common_name][i] = val
                                break
                        except:
                            continue
            except Exception as e:
                logger.warning(f"PHREEQC solution calculation failed for sample {i}: {e}")

        return si_results

    def validate_dissolution(
        self,
        si_dict: Dict[str, NDArray[np.floating]],
        mineral_names: List[str],
        si_threshold: float = 0.5
    ) -> NDArray[np.bool_]:
        """
        Check if dissolution is thermodynamically feasible.
        Feasible if SI < threshold (undersaturated or near equilibrium).
        
        Parameters
        ----------
        si_dict : dict
            Output from calculate_si.
        mineral_names : list of str
            Minerals identified by the inverter.
        si_threshold : float
            Threshold for supersaturation.
            
        Returns
        -------
        plausible : ndarray of shape (n_samples, n_minerals)
        """
        n_samples = list(si_dict.values())[0].shape[0]
        n_minerals = len(mineral_names)
        plausible = np.ones((n_samples, n_minerals), dtype=bool)

        for j, m_name in enumerate(mineral_names):
            if m_name in si_dict:
                # If SI > threshold, mineral is supersaturated and cannot dissolve
                # Mark as implausible
                plausible[:, j] = si_dict[m_name] <= si_threshold

        return plausible
