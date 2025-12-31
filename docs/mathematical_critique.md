# Mathematical Critique of NeutroHydro Model

**Date**: December 31, 2025
**Module**: `neutrohydro`

This document provides a critical mathematical review of the NeutroHydro framework, identifying potential limitations and proposing rigorous solutions.

## 1. The "Weighting Paradox" in Mineral Inversion (Addressed)

### The Issue

The original `MineralInverter` minimized the weighted norm:
$$ \min_{s \ge 0} \| D \cdot (c - A s) \|_2^2 $$
where the weights $D$ were derived from the Baseline Fraction $\pi_G$. This risked downweighting anthropogenic markers (perturbations).

### Resolution

The current implementation allows for **Unweighted Inversion** or **Targeted Weighting**. By default, the model now uses a balanced approach where critical ions (like NO3) are not penalized, ensuring that pollution signals are not ignored.

## 2. Mixing vs. Mineral Dissolution

### The Issue

The NNLS solver assumes:
$$ c_{total} = \sum (\text{Mineral}_k \times \text{Mass}_k) $$
This assumes all solutes come from dissolving solid phases.

* **Reality**: Groundwater often involves **Mixing** with a pre-existing brine (e.g., Seawater).
* **Approximation**: The model approximates Seawater as a sum of `Halite` + `Sylvite` + `Gypsum` + ...
* **Critique**: This loses the **Constant Proportion** constraint of Seawater (e.g., $Cl/Br$ ratio). It allows the model to "break" Seawater into separate salts, which is physically impossible in simple mixing.

### Mathematical Solution

Add **Fluid Endmembers** to the Stoichiometric Matrix $A$:

* Define a "mineral" called `Seawater` with the exact ionic composition of standard seawater.
* $$ A_{Seawater} = [Na=468, Mg=53, Ca=10, Cl=545, SO4=28, ...] $$
* This forces the solver to use the *exact* seawater ratio, improving validity for salinization studies.

## 3. Non-Uniqueness of Ion Exchange (Addressed via Thermodynamics)

### The Issue

I introduced `Exchanger` phases (e.g., $Ca \to 2Na$) to model ion exchange.

* **Mathematical Risk**: This increases **Multicollinearity**.
  * *Scenario*: High Na, Low Ca.
  * *Explanation A*: Dissolve Halite ($Na, Cl$) + Precipitate Calcite ($-Ca, -CO3$).
  * *Explanation B*: Ion Exchange ($Ca \to 2Na$).
* **Solver Behavior**: NNLS will pick the path of least resistance (lowest residual). It cannot distinguish between these mechanisms without isotopic data.

### Resolution

**Thermodynamic Validation**: The integration of PHREEQC-based Saturation Indices (SI) acts as a physical constraint. If Calcite is undersaturated, precipitation is disallowed, forcing the model to choose Ion Exchange if that is the only viable path. This significantly reduces non-uniqueness.

## 4. Error Propagation in NDG (Addressed via Robust PCA)

### The Issue

The NDG Encoder calculates $T, I, F$ sequentially.

* $T$ = Robust PCA.
* $I$ = PCA on Residuals.
* $F$ = Distance to $T+I$.
* **Critique**: Errors in the estimation of $T$ (e.g., wrong rank) propagate to $I$ and $F$. If $T$ overfits, $I$ and $F$ vanish.

### Resolution

**Robust PCA Default**: The framework now defaults to **Robust PCA** (RPCA) for baseline estimation. RPCA is mathematically designed to separate a low-rank matrix ($T$) from sparse errors ($F$) without overfitting, making the decomposition stable even without extensive cross-validation for rank.

## 5. The "Rule-Based" Override Problem (Addressed via Context-Awareness)

### The Issue

The integration of **WHO Quality Flags** and **Gibbs Constraints** introduces a rule-based logic layer on top of the optimization layer.

* **Scenario**: The NNLS solver wants to fit `Halite` to explain Cl. The Gibbs constraint says "Rock Dominance" and bans `Halite`. The WHO flag says "Saline Intrusion" and forces `Halite` back in.
* **Critique**: This creates a **Hybrid System** where the objective function is dynamically modified by discrete logic gates. This makes the model behavior non-smooth and potentially sensitive to the specific thresholds used in the rules.

### Resolution

**Context-Aware Inversion**: The system now prioritizes **Pollution Context** (WHO Flags) over **Geological Context** (Gibbs) when they conflict. This is a deliberate design choice: if a sample is demonstrably polluted (e.g., Cl > 1000 mg/L), the "Rock Dominance" assumption of the Gibbs diagram is invalid. This hierarchy resolves the conflict deterministically.

## 6. Simpson's Ratio (Revelle Coefficient) Discretization

### The Issue

The model uses discrete bins for Simpson's Ratio (e.g., "Moderately contaminated" vs "Injuriously contaminated").

* **Critique**: Discretization throws away information. A sample at an SR of 2.7 is labeled "Moderately", while 2.9 is "Injuriously", despite being chemically nearly identical.

### Mathematical Solution

**Continuous Scoring**: Use the raw ratio values (Standard and Freshening) for any downstream statistical analysis (like correlation or clustering), and reserve the discrete classes only for the final human-readable report.
