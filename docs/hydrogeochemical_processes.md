# Hydrogeochemical Processes in NeutroHydro

**Module**: `neutrohydro`

This document explains how specific hydrogeochemical processes (Mixing, Salinization, Ion Exchange) are mathematically represented within the NeutroHydro framework and how the model accounts for them.

## 1. Mixing and Salinization

**Process**: The physical mixing of two or more distinct water bodies (e.g., fresh recharge + saline connate water).

* **Mathematical Nature**: Linear. $C_{mix} = f_1 C_1 + f_2 C_2$.
* **Model Representation**: **Truth ($T$) Channel**.

### How it works

The **NDG Encoder** uses Robust PCA (or Low-Rank Approximation) to define the "Truth" baseline.

* The Principal Components (PCs) of the $T$ channel naturally align with the **Mixing Lines**.
* **Salinization** (e.g., Seawater Intrusion) typically appears as the **First Principal Component (PC1)** because it explains the largest variance in total dissolved solids (TDS).
* **Validity**: Since mixing is a linear operation, the linear algebra underlying the $T$ channel is the mathematically valid term for these processes.

### Diagnostic Indices

- **Simpson Ratio (Revelle Coefficient)**: Defined as $SR = Cl^- / (HCO3^- + CO3^{2-})$.
* **Classes**:
  * $< 0.5$: Good quality
  * $0.5 - 1.3$: Slightly contaminated
  * $1.3 - 2.8$: Moderately contaminated
  * $2.8 - 6.6$: Injuriously contaminated
  * $6.6 - 15.5$: Highly contaminated
  * $> 15.5$: Extremely contaminated (Seawater)
* **Freshening Ratio (FR)**: The inverse $1/SR$. Large FR indicates freshwater dominance.

## 2. Ion Exchange

**Process**: The adsorption of one ion onto a clay surface and the release of another (e.g., $Ca^{2+}$ adsorbs, $2Na^+$ releases).

* **Mathematical Nature**: Non-linear / Non-conservative relative to the mixing line.
* **Model Representation**: **Indeterminacy ($I$) and Falsity ($F$) Channels**.

### How it works

Ion exchange creates a deviation from the linear mixing trend defined in $T$.

* *Example*: In simple mixing, if $Cl^-$ increases, $Na^+$ should increase proportionally (Halite ratio).
* *Effect*: If Ion Exchange occurs, $Na^+$ increases *more* than expected, while $Ca^{2+}$ increases *less* (or decreases).
* **The "Term"**: The model captures this deviation in the **Falsity ($F$)** matrix.
  * $F_{Na} > 0$ (Positive perturbation: Excess Na)
  * $F_{Ca} > 0$ (Negative perturbation: Deficit Ca - note $F$ is magnitude, sign is in residual)

### Explicit Modeling (The "Exchanger Term")

To explicitly solve for the mass of ions exchanged during **Mineral Inversion**, we can introduce **Pseudo-Minerals** with negative stoichiometric coefficients.

**Mathematically Valid Term**:
$$ \text{Clay}_{Na \to Ca} : \quad +1 \text{ Ca}^{2+} \quad -2 \text{ Na}^+ $$

* **Interpretation**: This "mineral" adds Calcium to the water and *removes* Sodium.
* **Constraint**: Since the NNLS solver requires positive mass ($x \ge 0$), we define two directional exchangers:
    1. **Direct Exchange** (Freshening): $Ca^{2+} \to 2Na^+$ (Release Na, Remove Ca)
    2. **Reverse Exchange** (Intrusion): $2Na^+ \to Ca^{2+}$ (Release Ca, Remove Na)

### Diagnostic Indices

- **Base Exchange Index (BEX)**: $BEX = Na^+ + K^+ + Mg^{2+} - 1.0716 \cdot Cl^-$ (meq/L).
* **Interpretation**:
  * $BEX > 0$: Freshening trend.
  * $BEX < 0$: Salinization trend (Seawater influence).
  * $BEX \approx 0$: No clear signal (Simple mixing or dissolution).

## 3. Redox Processes (Denitrification, Sulfate Reduction)

**Process**: Biogeochemical removal of species (e.g., $NO_3^- \to N_2(g)$) or addition (Nitrification).

* **Mathematical Nature**: Mass loss (Sink) or Gain (Source).
* **Model Representation**: **Falsity ($F$)** and **Redox Phases**.

### How it works

These processes look like "missing mass" relative to the conservative baseline.

* **The "Term"**: A high Falsity score for Nitrate ($F_{NO3}$) combined with a low Truth value indicates depletion.
* **Validity**: The $F$ channel provides the statistical evidence of the process.

### Explicit Modeling (The "Sink Term")

Similar to Ion Exchange, we can explicitly solve for the mass lost to redox processes by introducing **Redox Phases** with negative stoichiometry.

**Mathematically Valid Term**:
$$ \text{Sink}_{Denit} : \quad -1 \text{ NO}_3^- \quad +1 \text{ HCO}_3^- $$

* **Interpretation**: This "mineral" removes Nitrate and adds Alkalinity (bicarbonate).
* **Constraint**: Since the NNLS solver requires positive mass ($x \ge 0$), a positive value for this sink phase ($x_{denit} > 0$) mathematically accounts for the *negative* residual of Nitrate.
  * Equation: $C_{final} = C_{mix} + x_{denit} \times (-1)$
  * If $C_{final} < C_{mix}$ (Depletion), then $x_{denit}$ must be positive.

**Available Redox Phases**:

1. **Denitrification**: Removes $NO_3^-$, adds $HCO_3^-$.
2. **Sulfate Reduction**: Removes $SO_4^{2-}$, adds $HCO_3^-$.
3. **Nitrification**: Adds $NO_3^-$ (Source).

### Detection vs. Assumption

The model **detects** the process; it does not assume it is present.

* **Candidate Approach**: The Redox phases are provided as *candidates* to the solver.
* **Selection Logic**: The NNLS solver will only assign a positive value to `Sink_Denitrification` if there is a **mass deficit** in Nitrate that cannot be explained by mixing or other minerals.
  * If the observed Nitrate matches the expected background, the solver sets the Denitrification term to **0**.
  * If the observed Nitrate is *lower* than expected (a deficit), the solver increases the Denitrification term to minimize the error.
* **Result**: The magnitude of the term ($x_{denit}$) represents the **calculated mass** of Nitrate lost to the process.

## Summary of Terms

| Process | Mathematical Term in Model | Validity |
| :--- | :--- | :--- |
| **Mixing** | $T$ (Truth Matrix) | Valid (Linear Algebra) |
| **Salinization** | $T$ (PC1) + `Halite` (Mineral) | Valid (Stoichiometry) |
| **Ion Exchange** | $F$ (Falsity Matrix) + `Exchanger` (Pseudo-Mineral) | Valid (Perturbation Theory) |
| **Redox** | $F$ (Falsity Matrix) | Valid (Outlier Detection) |
