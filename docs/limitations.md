# Model Limitations and Validity

**Module**: `neutrohydro`

## Overview

While NeutroHydro provides a mathematically rigorous framework for groundwater chemometrics, it is subject to inherent limitations common to inverse geochemical modeling and statistical learning. Understanding these limitations is crucial for the scientific defensibility of the results.

## 1. Stoichiometric Assumptions

### Limitation
The **Mineral Inversion** module relies on a library of fixed stoichiometric formulas (e.g., Calcite = $CaCO_3$).
*   **Reality**: Natural minerals often exist as solid solutions (e.g., Magnesian Calcite $Ca_{(1-x)}Mg_xCO_3$) or have impurities.
*   **Impact**: The model may slightly over- or under-estimate the mass of a specific phase if the real-world mineral deviates from the ideal formula.

### Addressing Validity
*   **Residual Analysis**: The model calculates a `residual_norm` for every sample. A high residual indicates that the standard library cannot fully explain the water chemistry, prompting a check for non-standard minerals.
*   **Endmember Expansion**: The "Scientific" library includes pure endmembers (e.g., Albite and Anorthite) rather than a generic "Plagioclase," allowing the model to mix them to approximate the real solid solution.

## 2. Non-Uniqueness of Inversion

### Limitation
The problem of reconstructing mineral assemblages from dissolved ions is mathematically **underdetermined** (more minerals than ions) or **non-unique**.
*   **Example**: Dissolved $Ca^{2+}$ and $SO_4^{2-}$ could come from Gypsum ($CaSO_4 \cdot 2H_2O$) or Anhydrite ($CaSO_4$). Chemically, they produce the exact same ions.

### Addressing Validity
*   **Parsimony Principle**: The Non-Negative Least Squares (NNLS) algorithm inherently favors "sparse" solutions, selecting the fewest minerals needed to explain the data.
*   **Contextual Validation**: The model outputs "Plausible Minerals." The user must validate if these make geological sense (e.g., Anhydrite is rare in shallow aquifers; Gypsum is more likely).
*   **Thermodynamic Consistency**: Results should be cross-referenced with Saturation Indices (SI) if pH and temperature data are available (external validation).

## 3. Linearity of Baseline (PCA/PLS)

### Limitation
The **NDG Encoder** and **PNPLS** use linear projections (PCA-based) to define the "Baseline" (Truth).
*   **Reality**: Natural geochemical evolution (e.g., redox fronts, sorption isotherms) can be highly non-linear.

### Addressing Validity
*   **Neutrosophic Compensation**: This is the core strength of NeutroHydro. Unlike standard PCA which forces non-linear outliers into the model (distorting the baseline), NeutroHydro captures non-linear deviations in the **Indeterminacy ($I$)** and **Falsity ($F$)** channels.
*   **Interpretation**: A high $I$ or $F$ score doesn't just mean "error"; it often signifies a non-linear geochemical process (like denitrification) that deviates from the linear mixing baseline.

## 4. Data Completeness (Missing Ions)

### Limitation
Geochemical inversion requires charge balance. Missing major ions (especially Alkalinity/HCO3) makes it impossible to distinguish certain minerals (e.g., Calcite vs. Gypsum).

### Addressing Validity
*   **Adaptive Filtering**: The model dynamically scans the input dataset. If a critical ion (e.g., Nitrate) is missing, all minerals requiring that ion (e.g., Nitrocalcite) are **automatically removed** from the candidate list.
*   **Benefit**: This prevents "ghost" minerals (hallucinations) and ensures that any identified mineral is positively supported by the available data.

## 5. "Closed System" Assumption

### Limitation
Mass balance inversion assumes that the dissolved ions come solely from mineral dissolution/precipitation within the control volume.
*   **Reality**: Groundwater is an open system. Ions can be added via rainfall, evaporation, or anthropogenic inputs.

### Addressing Validity
*   **Anthropogenic Markers**: We explicitly include "Pollution Proxies" (e.g., Niter, Nitrocalcite) in the library to account for non-geogenic inputs.
*   **Evaporation Handling**: Conservative ions (Cl, Br) are used in the baseline to track physical concentration effects (evaporation) separate from chemical reactions.

## 6. Heuristic Constraints (Gibbs, Simpson, WHO)

### Limitation
The model now incorporates empirical hydrogeochemical rules (Gibbs Diagram, Simpson's Ratio, WHO Guidelines) to constrain the mathematical inversion.
*   **Empirical Nature**: These rules are generalizations derived from global datasets. They may not hold in specific, complex local geologies (e.g., a "Rock Dominance" zone that happens to have a local salt deposit).
*   **Discontinuity**: The use of thresholds (e.g., Simpson Ratio > 15.5) creates discrete classification bins for what is physically a continuous variable.

### Addressing Validity
*   **Context-Aware Overrides**: The "Quality Flag" integration allows the model to override general rules (like Gibbs) if specific evidence (like WHO exceedances) points to a contradiction (e.g., Saline Intrusion).
*   **Transparency**: The model reports the calculated indices (`Simpson_Ratio`, `CAI`) alongside the mineral results, allowing the user to see *why* a constraint was applied.
