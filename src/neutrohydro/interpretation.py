
"""
Geochemical interpretation and insight generation.

This module provides "intelligent" analysis of the pipeline results,
identifying paradoxes (e.g., supersaturated dissolution), dominant processes,
and specific source attribution logic for key contaminants like Fluoride.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from neutrohydro.pipeline import PipelineResults
from neutrohydro.minerals import MineralInversionResult

@dataclass
class Insight:
    """A single geochemical insight or warning."""
    category: str  # "Paradox", "Process", "Source", "Warning"
    message: str
    samples: List[str] = field(default_factory=list)
    confidence: str = "Medium"  # Low, Medium, High

@dataclass
class InterpretationReport:
    """Container for all interpretation results."""
    insights: List[Insight]
    fluoride_source_summary: pd.DataFrame
    mineral_paradoxes: pd.DataFrame

class GeochemicalInterpreter:
    """
    Analyzes pipeline results to generate geochemical insights.
    """

    def __init__(
        self, 
        results: PipelineResults, 
        sample_codes: Optional[List[str]] = None,
        geology: Optional[str] = None
    ):
        self.results = results
        self.mineral_res = results.mineral_result
        self.n_samples = results.y_pred.shape[0]
        self.geology = geology if geology is not None else "Unknown/Mixed"
        
        if sample_codes is None:
            self.sample_codes = [f"Sample_{i+1}" for i in range(self.n_samples)]
        else:
            self.sample_codes = sample_codes

    def interpret(self) -> InterpretationReport:
        """Run all interpretation routines."""
        insights = []
        
        # 1. Check for Mineral Paradoxes (Dissolution of Supersaturated Minerals)
        paradox_df = self._check_mineral_paradoxes(insights)
        
        # 2. Analyze Ion Exchange (CAI)
        self._analyze_ion_exchange(insights)
        
        # 3. Specific Fluoride Source Analysis
        f_summary = self._analyze_fluoride_sources(insights)
        
        # 4. Geological Context Analysis
        if self.geology:
            self._analyze_geology(insights)
            
        # 5. Universal Framework Analysis
        self._analyze_redox(insights)
        self._analyze_pollution(insights)
        self._analyze_salinity(insights)
        
        return InterpretationReport(
            insights=insights,
            fluoride_source_summary=f_summary,
            mineral_paradoxes=paradox_df
        )

    def _check_mineral_paradoxes(self, insights: List[Insight]) -> pd.DataFrame:
        """
        Identify samples where the model predicts mineral contribution 
        but the water is supersaturated (SI > 0).
        """
        if not self.mineral_res or not self.mineral_res.saturation_indices:
            return pd.DataFrame()

        paradoxes = []
        
        # Threshold for "significant" contribution (e.g., > 0.1 meq/L)
        contrib_threshold = 0.1
        # Threshold for "significant" supersaturation (e.g., SI > 0.1)
        si_threshold = 0.1

        mineral_names = self.mineral_res.mineral_names
        s_matrix = self.mineral_res.s  # (n_samples, n_minerals)

        for i, sample in enumerate(self.sample_codes):
            for j, mineral in enumerate(mineral_names):
                # Get contribution
                contrib = s_matrix[i, j]
                
                # Get SI
                # Note: mineral names in s_matrix might match keys in saturation_indices
                # We need to handle potential naming mismatches if they exist
                # Assuming exact match for now based on minerals.py
                if mineral in self.mineral_res.saturation_indices:
                    si = self.mineral_res.saturation_indices[mineral][i]
                    
                    if contrib > contrib_threshold and si > si_threshold:
                        paradoxes.append({
                            "Sample": sample,
                            "Mineral": mineral,
                            "Contribution": contrib,
                            "SI": si,
                            "Type": "Supersaturation Paradox"
                        })

        df = pd.DataFrame(paradoxes)
        
        if not df.empty:
            # Group by mineral to generate insights
            for mineral, group in df.groupby("Mineral"):
                count = len(group)
                msg = (f"Mineral '{mineral}' is predicted as a source in {count} samples "
                       f"where it is thermodynamically supersaturated (SI > {si_threshold}). "
                       "This suggests the mineral is NOT dissolving, but rather precipitating "
                       "or the water is in a metastable state. Consider other sources.")
                insights.append(Insight(
                    category="Paradox",
                    message=msg,
                    samples=group["Sample"].tolist(),
                    confidence="High"
                ))
                
        return df

    def _analyze_ion_exchange(self, insights: List[Insight]):
        """Analyze Chloro-Alkaline Indices."""
        # We need CAI indices. If they are not in mineral_res.indices, we can't do much.
        # However, minerals.py calculates them if we updated it.
        # Let's assume they are there or we skip.
        if not self.mineral_res or not hasattr(self.mineral_res, 'indices') or not self.mineral_res.indices:
            return

        cai1 = self.mineral_res.indices.get("CAI_1")
        if cai1 is None:
            return

        # Count samples
        normal_exchange = np.sum(cai1 < -0.1)  # CAI < 0
        reverse_exchange = np.sum(cai1 > 0.1)  # CAI > 0
        
        if normal_exchange > self.n_samples * 0.5:
            insights.append(Insight(
                category="Process",
                message=f"Dominant process is Normal Ion Exchange (Freshening). "
                        f"{normal_exchange}/{self.n_samples} samples show Na release and Ca/Mg uptake.",
                confidence="High"
            ))
        elif reverse_exchange > self.n_samples * 0.5:
            insights.append(Insight(
                category="Process",
                message=f"Dominant process is Reverse Ion Exchange (Salinization). "
                        f"{reverse_exchange}/{self.n_samples} samples show Ca/Mg release and Na uptake.",
                confidence="High"
            ))

    def _analyze_fluoride_sources(self, insights: List[Insight]) -> pd.DataFrame:
        """
        Specific logic for Fluoride attribution.
        """
        if not self.mineral_res or not self.mineral_res.saturation_indices or "Fluorite" not in self.mineral_res.saturation_indices:
            return pd.DataFrame()

        si_fluorite = self.mineral_res.saturation_indices["Fluorite"]
        
        # Check for CAI-1 to infer Ca removal
        cai1 = self.mineral_res.indices.get("CAI_1") if self.mineral_res.indices else None
        
        classifications = []
        
        for i, sample in enumerate(self.sample_codes):
            si = si_fluorite[i]
            cai_val = cai1[i] if cai1 is not None else 0
            
            if si > 0.1:
                status = "Supersaturated (Not Fluorite Dissolution)"
                if cai_val < -0.1:
                    mech = "Silicate Weathering + Ca-Removal (Ion Exchange)"
                else:
                    mech = "Desorption / Silicate Weathering / Anthropogenic"
            elif si < -0.5:
                status = "Undersaturated"
                mech = "Fluorite Dissolution Possible"
            else:
                status = "Equilibrium"
                mech = "Solubility Control"
                
            classifications.append({
                "Sample": sample,
                "Fluorite_SI": si,
                "Status": status,
                "Likely_Mechanism": mech
            })
            
        df = pd.DataFrame(classifications)
        
        # Generate summary insight
        supersat_count = np.sum(si_fluorite > 0.1)
        if supersat_count > 0:
            # Check if Ion Exchange is dominant in these samples
            supersat_indices = np.where(si_fluorite > 0.1)[0]
            if cai1 is not None:
                exchange_count = np.sum(cai1[supersat_indices] < -0.1)
                if exchange_count > len(supersat_indices) * 0.5:
                    msg = (f"Fluoride is supersaturated in {supersat_count} samples. "
                           "Strong evidence of 'Silicate Weathering + Ca-Removal' mechanism: "
                           "High F is maintained by Ion Exchange removing Calcium, preventing Fluorite precipitation.")
                else:
                    msg = (f"Fluorite is supersaturated in {supersat_count} samples. "
                           "High Fluoride is likely driven by pH-dependent desorption or silicate weathering.")
            else:
                msg = (f"Fluorite is supersaturated in {supersat_count} samples. "
                       "High Fluoride is likely driven by pH-dependent desorption or silicate weathering.")

            insights.append(Insight(
                category="Source",
                message=msg,
                samples=df[df["Fluorite_SI"] > 0.1]["Sample"].tolist(),
                confidence="High"
            ))
            
        return df

    def _analyze_geology(self, insights: List[Insight]):
        """
        Refine insights based on geological context.
        """
        geo = self.geology.lower()
        
        # 1. Crystalline / Hard Rock
        if "granite" in geo or "gneiss" in geo:
            insights.append(Insight(
                category="Geology",
                message=f"Geology '{self.geology}' supports Silicate Weathering (Biotite/Feldspar) as a primary source of ions (F, Na, K).",
                confidence="High"
            ))
        elif "basalt" in geo or "volcanic" in geo:
             insights.append(Insight(
                category="Geology",
                message=f"Geology '{self.geology}' supports Silicate Weathering of ferromagnesian minerals (Plagioclase/Pyroxene), likely contributing Ca, Mg, and Si.",
                confidence="High"
            ))
        elif "metamorphic" in geo:
             insights.append(Insight(
                category="Geology",
                message=f"Geology '{self.geology}' suggests variable mineralogy. Look for fracture-controlled flow and water-rock interaction signatures.",
                confidence="Medium"
            ))

        # 2. Sedimentary
        elif "carbonate" in geo or "limestone" in geo or "karst" in geo:
            insights.append(Insight(
                category="Geology",
                message=f"Geology '{self.geology}' strongly supports Carbonate Dissolution (Calcite/Dolomite). Expect Ca-HCO3 or Mg-HCO3 water types.",
                confidence="High"
            ))
        elif "sandstone" in geo or "shale" in geo or "clastic" in geo:
            insights.append(Insight(
                category="Geology",
                message=f"Geology '{self.geology}' supports Ion Exchange processes (clay minerals) and Silicate Weathering. Expect Na-HCO3 or mixed water types.",
                confidence="High"
            ))

        # 3. Unconsolidated / Special
        elif "coastal" in geo or "marine" in geo:
            insights.append(Insight(
                category="Geology",
                message=f"Geology '{self.geology}' requires checking for Seawater Intrusion (Na-Cl type, high TDS).",
                confidence="High"
            ))
            # Add specific check for Saline Intrusion if data supports it
            # (Simple check: Na/Cl ratio ~ 0.8-1.0 and high Cl)
            # This would be an enhancement to the logic.
            
        elif "alluvial" in geo or "valley" in geo:
            insights.append(Insight(
                category="Geology",
                message=f"Geology '{self.geology}' suggests porous media flow with high residence times. Expect mixing and potential evaporation signatures.",
                confidence="Medium"
            ))
            
        elif "anthropogenic" in geo or "urban" in geo:
            insights.append(Insight(
                category="Geology",
                message=f"Setting '{self.geology}' implies high risk of contamination. Check Nitrate (NO3) and Sulfate (SO4) levels.",
                confidence="High"
            ))

    def _analyze_redox(self, insights: List[Insight]):
        """Analyze Redox Zonation."""
        if not self.mineral_res or self.mineral_res.redox_state is None:
            return

        states = self.mineral_res.redox_state
        n_oxic = states.count("Oxic")
        n_anoxic = states.count("Anoxic")
        n_suboxic = states.count("Suboxic/Mixed")
        
        msg = f"Redox Zonation: {n_oxic} Oxic, {n_suboxic} Suboxic, {n_anoxic} Anoxic samples."
        
        if n_anoxic > 0:
            msg += " Anoxic conditions detected (High Fe/Mn). Risk of Arsenic mobilization if geology contains As-bearing sulfides."
            insights.append(Insight(
                category="Redox",
                message=msg,
                samples=[self.sample_codes[i] for i, s in enumerate(states) if s == "Anoxic"],
                confidence="High"
            ))
        else:
            insights.append(Insight(
                category="Redox",
                message=msg,
                confidence="Medium"
            ))

    def _analyze_pollution(self, insights: List[Insight]):
        """Analyze Pollution Fingerprints."""
        if not self.mineral_res or self.mineral_res.pollution_indices is None:
            return
            
        indices = self.mineral_res.pollution_indices
        no3_cl = indices.get("no3_cl_ratio")
        k_na = indices.get("k_na_ratio")
        no3_meq = indices.get("no3_meq")
        cl_meq = indices.get("cl_meq")
        
        if no3_cl is None or k_na is None:
            return
            
        # 1. Chemical N-Fertilizers (Very High NO3/Cl > 1.0)
        # Requires significant NO3 presence (> 0.1 meq/L ~ 6 mg/L) to avoid false positives in pristine water
        mask_chem_fert = (no3_cl > 1.0) & (no3_meq > 0.1)
        n_chem_fert = np.sum(mask_chem_fert)
        
        if n_chem_fert > 0:
            samples = [self.sample_codes[i] for i in range(self.n_samples) if mask_chem_fert[i]]
            insights.append(Insight(
                category="Pollution",
                message=f"Chemical N-Fertilizer signature detected in {n_chem_fert} samples (NO3/Cl > 1.0).",
                samples=samples,
                confidence="Medium"
            ))

        # 2. Sewage/Manure (Moderate NO3/Cl: 0.05 - 1.0) AND High Cl
        # Sewage implies high Cl background.
        # Thresholds: NO3/Cl between 0.05 and 1.0, and Cl > 1.0 meq/L (~35 mg/L)
        mask_sewage = (no3_cl > 0.05) & (no3_cl <= 1.0) & (cl_meq > 1.0)
        n_sewage = np.sum(mask_sewage)
        
        if n_sewage > 0:
            samples = [self.sample_codes[i] for i in range(self.n_samples) if mask_sewage[i]]
            insights.append(Insight(
                category="Pollution",
                message=f"Potential Sewage/Manure contamination detected in {n_sewage} samples (0.05 < NO3/Cl < 1.0 with High Cl).",
                samples=samples,
                confidence="Medium"
            ))
            
        # 3. Potash Fertilizer (High K/Na)
        # Threshold > 0.2 (approx, depends on geology)
        # Note: K-feldspar weathering also gives K, but usually K < Na.
        mask_fert = k_na > 0.2
        n_fert = np.sum(mask_fert)
        
        if n_fert > 0:
            samples = [self.sample_codes[i] for i in range(self.n_samples) if mask_fert[i]]
            insights.append(Insight(
                category="Pollution",
                message=f"Potential Potash Fertilizer impact detected in {n_fert} samples (K/Na > 0.2). Verify if agriculture is present.",
                samples=samples,
                confidence="Low"  # Lower confidence due to geological K overlap
            ))

    def _analyze_salinity(self, insights: List[Insight]):
        """Analyze Salinity Origin."""
        if not self.mineral_res or self.mineral_res.salinity_indices is None:
            return
            
        indices = self.mineral_res.salinity_indices
        ri = indices.get("revelle_index")
        na_cl = indices.get("na_cl_ratio")
        
        if ri is None or na_cl is None:
            return
            
        # 1. Seawater Intrusion (RI > 1 AND Na/Cl < 0.86)
        mask_swi = (ri > 1.0) & (na_cl < 0.86)
        n_swi = np.sum(mask_swi)
        
        if n_swi > 0:
            samples = [self.sample_codes[i] for i in range(self.n_samples) if mask_swi[i]]
            insights.append(Insight(
                category="Salinity",
                message=f"Seawater Intrusion signals detected in {n_swi} samples (Revelle Index > 1, Na/Cl < 0.86).",
                samples=samples,
                confidence="High"
            ))
            
        # 2. Halite Dissolution (Na/Cl ~ 1)
        # Range 0.86 - 1.2
        mask_halite = (na_cl >= 0.86) & (na_cl <= 1.2)
        n_halite = np.sum(mask_halite)
        
        if n_halite > 0:
             insights.append(Insight(
                category="Salinity",
                message=f"Halite Dissolution likely in {n_halite} samples (Na/Cl ~ 1).",
                confidence="Medium"
            ))
