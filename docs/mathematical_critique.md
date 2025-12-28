# Mathematical Critique of NeutroHydro Model

**Date**: December 28, 2025
**Module**: `neutrohydro`

This document provides a critical mathematical review of the NeutroHydro framework, identifying potential limitations and proposing rigorous solutions.

## 1. The "Weighting Paradox" in Mineral Inversion

### The Issue
The current `MineralInverter` minimizes the weighted norm:
$$ \min_{s \ge 0} \| D \cdot (c - A s) \|_2^2 $$
where the weights $D$ are derived from the Baseline Fraction $\pi_G$:
$$ D_{jj} \approx (\pi_G(j))^\eta $$

*   **Logic**: Trust the "Baseline" ions more; downweight the "Perturbed/Noisy" ions.
*   **Consequence**: Anthropogenic markers (e.g., Nitrate from fertilizer) are often **perturbations** (High $F$, Low $\pi_G$).
*   **The Paradox**: By downweighting the perturbation, the solver is effectively told "It is okay to ignore Nitrate."
    *   **Result**: The model may underestimate the mass of `Nitrocalcite` or `Niter` because the penalty for missing the Nitrate target is small.

### Mathematical Solution
For **Forensic Analysis** (identifying pollution), the weighting scheme should be inverted or removed:
1.  **Unweighted Inversion**: Set $D = I$ (Identity). This forces the model to explain *all* ions, including pollutants.
2.  **Targeted Weighting**: Explicitly set high weights for suspected markers (e.g., $D_{NO3} = 1.0$) regardless of their $\pi_G$ score.

## 2. Mixing vs. Mineral Dissolution

### The Issue
The NNLS solver assumes:
$$ c_{total} = \sum (\text{Mineral}_k \times \text{Mass}_k) $$
This assumes all solutes come from dissolving solid phases.
*   **Reality**: Groundwater often involves **Mixing** with a pre-existing brine (e.g., Seawater).
*   **Approximation**: The model approximates Seawater as a sum of `Halite` + `Sylvite` + `Gypsum` + ...
*   **Critique**: This loses the **Constant Proportion** constraint of Seawater (e.g., $Cl/Br$ ratio). It allows the model to "break" Seawater into separate salts, which is physically impossible in simple mixing.

### Mathematical Solution
Add **Fluid Endmembers** to the Stoichiometric Matrix $A$:
*   Define a "mineral" called `Seawater` with the exact ionic composition of standard seawater.
*   $$ A_{Seawater} = [Na=468, Mg=53, Ca=10, Cl=545, SO4=28, ...] $$
*   This forces the solver to use the *exact* seawater ratio, improving validity for salinization studies.

## 3. Non-Uniqueness of Ion Exchange

### The Issue
I introduced `Exchanger` phases (e.g., $Ca \to 2Na$) to model ion exchange.
*   **Mathematical Risk**: This increases **Multicollinearity**.
    *   *Scenario*: High Na, Low Ca.
    *   *Explanation A*: Dissolve Halite ($Na, Cl$) + Precipitate Calcite ($-Ca, -CO3$).
    *   *Explanation B*: Ion Exchange ($Ca \to 2Na$).
*   **Solver Behavior**: NNLS will pick the path of least resistance (lowest residual). It cannot distinguish between these mechanisms without isotopic data.

### Mathematical Solution
**Regularization**: Apply L2 (Ridge) or L1 (Lasso) penalties to the Exchanger terms to ensure they are only selected when standard minerals *cannot* explain the data (i.e., when Cl is conservative but Na is not).

## 4. Error Propagation in NDG

### The Issue
The NDG Encoder calculates $T, I, F$ sequentially.
*   $T$ = Robust PCA.
*   $I$ = PCA on Residuals.
*   $F$ = Distance to $T+I$.
*   **Critique**: Errors in the estimation of $T$ (e.g., wrong rank) propagate to $I$ and $F$. If $T$ overfits, $I$ and $F$ vanish.

### Mathematical Solution
**Cross-Validation**: The rank of $T$ (number of components) must be selected via Cross-Validation ($Q^2$ metric) to ensure $T$ only captures the stable baseline, leaving the true noise/perturbation for $I$ and $F$.

## 5. The "Rule-Based" Override Problem

### The Issue
The integration of **WHO Quality Flags** and **Gibbs Constraints** introduces a rule-based logic layer on top of the optimization layer.
*   **Scenario**: The NNLS solver wants to fit `Halite` to explain Cl. The Gibbs constraint says "Rock Dominance" and bans `Halite`. The WHO flag says "Saline Intrusion" and forces `Halite` back in.
*   **Critique**: This creates a **Hybrid System** where the objective function is dynamically modified by discrete logic gates. This makes the model behavior non-smooth and potentially sensitive to the specific thresholds used in the rules.

### Mathematical Solution
**Soft Constraints / Priors**: Instead of binary Banning/Forcing, these heuristics should be implemented as **Bayesian Priors**.
*   Gibbs "Rock Dominance" $\to$ Low Prior Probability for Halite.
*   WHO "Saline Intrusion" $\to$ High Prior Probability for Halite.
*   This allows the data (Likelihood) to still have a say, rather than being overruled by a hard logic gate.

## 6. Simpson's Ratio Discretization

### The Issue
The model uses discrete bins for Simpson's Ratio (e.g., "Moderately Saline" vs "Highly Saline").
*   **Critique**: Discretization throws away information. A sample at 2.7 is labeled "Moderately", while 2.9 is "Highly", despite being chemically nearly identical.

### Mathematical Solution
**Continuous Scoring**: Use the raw ratio values (Standard and Inverse) for any downstream statistical analysis (like correlation or clustering), and reserve the discrete classes only for the final human-readable report.
