# Mineral Stoichiometric Inversion

**Module**: `neutrohydro.minerals`

## Overview

The mineral inference module uses **stoichiometric inversion** to estimate plausible mineral contributions from ion concentration data. This provides a **geochemical interpretation** of water composition in terms of **mineral dissolution/weathering** sources.

**Key innovation**: Uses baseline fractions $π_G$ to **weight ions**, emphasizing baseline-dominant ions in the inversion.

## Mathematical Foundation

### 1. Stoichiometric Model

#### 1.1 Forward Model

Ion concentrations arise from **mineral dissolution**:

$$c = A s + r$$

where:

- $c \in \mathbb{R}^m$: Observed ion concentrations (in **meq/L**, recommended)
- $A \in \mathbb{R}^{m \times K}$: **Stoichiometric matrix** (mineral compositions)
- $s \in \mathbb{R}^K_{\geq 0}$: **Mineral contributions** (non-negative)
- $r \in \mathbb{R}^m$: **Residual** (unmodeled processes)

**Units**:

- $c_\ell$ in meq/L (milli-equivalents per liter) ensures **charge balance**
- $s_k$ in arbitrary units (relative contributions)
- $A_{\ell k}$ = meq of ion $\ell$ per unit mineral $k$

#### 1.2 Stoichiometric Matrix

Column $k$ of $A$ encodes the composition of mineral $k$:

$$A_{:,k} = \begin{bmatrix} a_{1k} \\ \vdots \\ a_{mk} \end{bmatrix}$$

where $a_{\ell k}$ = equivalents of ion $\ell$ produced per unit dissolution of mineral $k$.

**Example** (Halite, NaCl):

$$\text{NaCl} \to \text{Na}^+ + \text{Cl}^-$$

In meq/L, dissolution of 1 mole NaCl produces:

- 1 meq Na⁺
- 1 meq Cl⁻

So for halite column:

```
Ion order: [Ca²⁺, Mg²⁺, Na⁺, K⁺, HCO₃⁻, Cl⁻, SO₄²⁻, NO₃⁻, F⁻]
Halite:    [0,    0,    1.0, 0,   0,      1.0, 0,     0,     0  ]
```

See Section 4 for standard mineral library.

### 2. Inverse Problem

#### 2.1 Unweighted NNLS

$$\hat{s} = \arg\min_{s \geq 0} \|c - As\|^2$$

**Non-negative least squares (NNLS)**:

- Ensures $s_k \geq 0$ (minerals can only dissolve, not precipitate)
- Convex optimization (unique solution)
- Fast algorithms available (SciPy, active set method)

#### 2.2 Weighted NNLS (NeutroHydro Innovation)

Use baseline fractions $π_G$ to **emphasize baseline-dominant ions**:

$$\hat{s} = \arg\min_{s \geq 0} \|D(c - As)\|^2$$

where $D = \text{diag}(d_1, \ldots, d_m)$ and:

$$d_\ell = \pi_G(\text{ion}_\ell)^\eta$$

**Hyperparameters**:

- $\eta \geq 1$: Weighting exponent (default: 1.0)
  - $\eta = 1$: Linear weighting
  - $\eta > 1$: Stronger emphasis on baseline

**Rationale**: Baseline-dominant ions reflect **natural geochemical processes** (mineral weathering), while perturbation-dominant ions may reflect **anthropogenic inputs** (fertilizers, contamination, ion exchange) not modeled by simple dissolution.

#### 2.3 Equivalence to Standard NNLS

Weighted NNLS is equivalent to:

$$\hat{s} = \arg\min_{s \geq 0} \|c_w - A_w s\|^2$$

where:
$$c_w = D c, \quad A_w = D A$$

**Implementation**: Transform data, solve standard NNLS.

##### Proof of Equivalence

**Theorem**: The weighted NNLS problem:
$$\min_{s \geq 0} \|D(c - As)\|^2$$

is equivalent to the standard NNLS problem:
$$\min_{s \geq 0} \|c_w - A_w s\|^2$$

where $c_w = Dc$ and $A_w = DA$.

**Proof**:

Expand the weighted objective:
$$J(s) = \|D(c - As)\|^2$$

$$= (c - As)^T D^T D (c - As)$$

Since $D = \text{diag}(d_1, \ldots, d_m)$ is diagonal:
$$D^T D = D^2 = \text{diag}(d_1^2, \ldots, d_m^2)$$

Therefore:
$$J(s) = (c - As)^T D^2 (c - As)$$

$$= \sum_{\ell=1}^m d_\ell^2 (c_\ell - (As)_\ell)^2$$

Now consider the transformed problem with $c_w = Dc$ and $A_w = DA$:
$$J_w(s) = \|c_w - A_w s\|^2$$

$$= \|Dc - DAs\|^2$$

$$= \|D(c - As)\|^2$$

$$= J(s)$$

Since the objectives are **identical** for all $s$, they have the same minimizer:
$$\arg\min_{s \geq 0} J(s) = \arg\min_{s \geq 0} J_w(s)$$

$\square$

**Computational advantage**: Standard NNLS solvers (e.g., `scipy.optimize.nnls`) can be used directly after pre-multiplying by $D$.

### 3. Residual Diagnostics

#### 3.1 Weighted Residual

$$r = c - A\hat{s}$$

$$\|r\|_D = \|D r\| = \sqrt{\sum_{\ell=1}^m d_\ell^2 r_\ell^2}$$

**Interpretation**: Goodness of fit, accounting for ion weights.

#### 3.2 Plausibility Criteria

Mineral $k$ is **plausible** in sample $i$ if:

1. **Sufficient contribution**: $\hat{s}_{ik} > \tau_s$ (default: 0.01)
2. **Good fit**: $\|r_i\|_D \leq \tau_r$ (default: 1.0)

**Both** criteria must be satisfied.

**Rationale**:

- Criterion 1: Avoids spurious tiny contributions
- Criterion 2: Ensures model fits the data

### 4. Standard Mineral Library

NeutroHydro includes an expanded "Scientific Research Grade" library of **24 minerals/endmembers**, covering silicates, carbonates, evaporites, and specific anthropogenic markers.

#### 4.1 Natural Minerals (Geogenic)

| Mineral | Formula | Key Ions | Description |
|---------|---------|----------|-------------|
| **Calcite** | CaCO₃ | Ca²⁺, HCO₃⁻ | Carbonate dissolution |
| **Dolomite** | CaMg(CO₃)₂ | Ca²⁺, Mg²⁺, HCO₃⁻ | Carbonate dissolution |
| **Magnesite** | MgCO₃ | Mg²⁺, HCO₃⁻ | Magnesium carbonate |
| **Gypsum** | CaSO₄·2H₂O | Ca²⁺, SO₄²⁻ | Sulfate dissolution |
| **Anhydrite** | CaSO₄ | Ca²⁺, SO₄²⁻ | Anhydrous sulfate |
| **Halite** | NaCl | Na⁺, Cl⁻ | Saline deposits/intrusion |
| **Sylvite** | KCl | K⁺, Cl⁻ | Potash deposits |
| **Mirabilite** | Na₂SO₄·10H₂O | Na⁺, SO₄²⁻ | Sodium sulfate |
| **Thenardite** | Na₂SO₄ | Na⁺, SO₄²⁻ | Anhydrous sodium sulfate |
| **Glauberite** | Na₂Ca(SO₄)₂ | Na⁺, Ca²⁺, SO₄²⁻ | Mixed sulfate |
| **Epsomite** | MgSO₄·7H₂O | Mg²⁺, SO₄²⁻ | Magnesium sulfate |
| **Fluorite** | CaF₂ | Ca²⁺, F⁻ | Fluoride source |
| **Albite** | NaAlSi₃O₈ | Na⁺, HCO₃⁻ | Plagioclase weathering |
| **Anorthite** | CaAl₂Si₂O₈ | Ca²⁺, HCO₃⁻ | Plagioclase weathering |
| **K-feldspar** | KAlSi₃O₈ | K⁺, HCO₃⁻ | Orthoclase weathering |
| **Biotite** | K(Mg,Fe)₃... | K⁺, Mg²⁺, HCO₃⁻ | Mica weathering |

#### 4.2 Anthropogenic Markers (Pollution Proxies)

These phases are used to fingerprint specific contamination sources.

| Marker | Formula | Key Ions | Interpretation |
|--------|---------|----------|----------------|
| **Niter** | KNO₃ | K⁺, NO₃⁻ | Potassium-based fertilizers |
| **Soda Niter** | NaNO₃ | Na⁺, NO₃⁻ | Sodium-based fertilizers |
| **Nitrocalcite** | Ca(NO₃)₂ | Ca²⁺, NO₃⁻ | Calcium nitrate fertilizers |
| **Otavite** | CdCO₃ | Cd²⁺, HCO₃⁻ | Cadmium impurity marker |
| **Smithsonite** | ZnCO₃ | Zn²⁺, HCO₃⁻ | Industrial/Sewage marker |
| **Cerussite** | PbCO₃ | Pb²⁺, HCO₃⁻ | Industrial/Road marker |
| **Borax** | Na₂B₄O₇ | Na⁺, B | Detergent/Wastewater marker |
| **Malachite** | Cu₂CO₃(OH)₂ | Cu²⁺, HCO₃⁻ | Pesticide/Industrial marker |

#### 4.3 Redox Phases (Optional)

These phases represent biogeochemical sinks (mass loss) or sources (mass gain) driven by redox reactions. They are not enabled by default but can be added for advanced modeling.

| Phase | Formula | Key Ions | Interpretation |
|-------|---------|----------|----------------|
| **Sink_Denitrification** | NO₃⁻ → N₂ | NO₃⁻ (-1), HCO₃⁻ (+1) | Nitrate reduction (Sink) |
| **Sink_SulfateReduction** | SO₄²⁻ → H₂S | SO₄²⁻ (-1), HCO₃⁻ (+1) | Sulfate reduction (Sink) |
| **Source_Nitrification** | NH₄⁺ → NO₃⁻ | NO₃⁻ (+1) | Nitrification (Source) |

**Note**: Negative stoichiometry allows the NNLS solver (which finds positive coefficients) to account for mass loss (negative residuals).

**Note on Trace Metals**: The model includes markers for Cd, Zn, Pb, B, Cu, As, Cr, and U. However, these are **only evaluated if the corresponding ion data is provided**. If metal concentrations are missing, these minerals are automatically excluded from the inversion to prevent "ghost" assignments.

**Full Ion Order** (m = 17):

```
[Ca²⁺, Mg²⁺, Na⁺, K⁺, HCO₃⁻, Cl⁻, SO₄²⁻, NO₃⁻, F⁻, 
 Zn²⁺, Cd²⁺, Pb²⁺, B, Cu²⁺, As, Cr, U]
```

### 5. Adaptive Ion Handling

The `MineralInverter` and pipeline are designed to handle datasets with varying ion availability:

1. **Data-Driven Mineral Selection**: The model checks which ions are present in the input dataset.
2. **Automatic Filtering**: Any mineral requiring a missing ion is removed from the candidate list.
    - *Example*: If `Cd` is not measured, `Otavite` is removed.
    - *Example*: If `NO3` is not measured, `Niter`, `SodaNiter`, and `Nitrocalcite` are removed.
3. **Robustness**: This ensures that the model never "hallucinates" a mineral contribution based on missing data, while still allowing for sophisticated forensic analysis when comprehensive data is available.

### 6. Algorithm

#### Fitting (Per Sample)

**Input**:

- Ion vector $c_i \in \mathbb{R}^m$ (in meq/L)
- Stoichiometric matrix $A \in \mathbb{R}^{m \times K}$
- Baseline fractions $\pi_G \in \mathbb{R}^m$ (optional)
- Hyperparameters $\eta, \tau_s, \tau_r$

**Output**:

- Mineral contributions $\hat{s}_i \in \mathbb{R}^K$
- Residual $r_i \in \mathbb{R}^m$
- Plausibility mask $P_i \in \{0, 1\}^K$

```python
# 1. Compute weights
if pi_G is None:
    D = np.ones(m)
else:
    D = pi_G ** eta

# 2. Transform system
c_w = D * c_i
A_w = D[:, np.newaxis] * A

# 3. Solve NNLS
s_hat, rnorm = scipy.optimize.nnls(A_w, c_w)

# 4. Compute residual (original space)
r = c_i - A @ s_hat
r_norm_weighted = np.linalg.norm(D * r)

# 5. Plausibility
plausible = (s_hat > tau_s) & (r_norm_weighted <= tau_r)

# 6. Normalized fractions
s_total = s_hat.sum() + eps
mineral_fractions = s_hat / s_total

return s_hat, r, r_norm_weighted, plausible, mineral_fractions
```

### 6. Custom Minerals

Users can define custom mineral dictionaries:

```python
custom_minerals = {
    "MyMineral": {
        "formula": "XYZ",
        "stoichiometry": {
            "Ca2+": 2.0,  # meq per unit dissolution
            "SO42-": 2.0,
        },
        "description": "Custom mineral description"
    },
}

from neutrohydro.minerals import MineralInverter

inverter = MineralInverter(minerals=custom_minerals)
result = inverter.invert(c_meq, pi_G)
```

**Requirements**:

- Keys must match **ion order** (see `STANDARD_IONS`)
- Values in **meq per unit** (charge-consistent)
- Unlisted ions assumed zero contribution

### 7. Unit Conversion

#### 7.1 mg/L to meq/L

$$\text{meq/L} = \frac{\text{mg/L}}{M} \times |z|$$

where:

- $M$ = molar mass (g/mol)
- $|z|$ = absolute charge

**Example** (Ca²⁺):

- Molar mass = 40.078 g/mol
- Charge = +2
- 100 mg/L Ca²⁺ = $\frac{100}{40.078} \times 2 = 4.99$ meq/L

**Function provided**:

```python
from neutrohydro.minerals import convert_to_meq, ION_MASSES, ION_CHARGES

c_meq = convert_to_meq(
    concentrations_mg,
    ion_charges=ION_CHARGES,
    ion_masses=ION_MASSES,
    from_unit="mg/L"
)
```

#### 7.2 mmol/L to meq/L

$$\text{meq/L} = \text{mmol/L} \times |z|$$

**Example** (SO₄²⁻):

- 5 mmol/L SO₄²⁻ = $5 \times 2 = 10$ meq/L

### 8. Limitations and Caveats

#### 8.1 Model Assumptions

The stoichiometric model assumes:

1. **Simple dissolution**: Minerals dissolve incongruently
2. **No ion exchange**: Cations don't swap on clays
3. **No redox**: Oxidation states don't change (e.g., Fe²⁺ → Fe³⁺)
4. **No precipitation**: Minerals only dissolve, not precipitate
5. **Equilibrium**: Instantaneous reactions

**Violations common in real systems**:

- Cation exchange (Ca²⁺ ↔ 2Na⁺ on clays)
- Redox (O₂, NO₃⁻ reduction)
- Kinetics (slow weathering)
- Mixing (groundwater from multiple sources)

#### 8.2 Residual Sources

Large residuals ($\|r\|_D > \tau_r$) may indicate:

- **Missing endmembers**: Important minerals not in library
- **Non-mineral sources**: Fertilizers (NO₃⁻, K⁺), wastewater (Na⁺, Cl⁻)
- **Ion exchange**: Alters Ca/Na ratio without mineral dissolution
- **Atmospheric inputs**: Sea salt spray (Na⁺, Cl⁻, Mg²⁺, SO₄²⁻)
- **Data quality**: Measurement errors, charge imbalance

#### 8.3 Non-Uniqueness

NNLS solution may not be **unique** if:

- Minerals have similar stoichiometry (e.g., gypsum vs. anhydrite)
- System is underdetermined ($K > m$)

**Implication**: Interpret $\hat{s}$ as **one plausible decomposition**, not absolute truth.

### 9. Diagnostics

#### 9.1 Charge Balance

Check total cations vs. anions:

$$\sum_{\ell \in \text{cations}} c_\ell \approx \sum_{\ell \in \text{anions}} c_\ell$$

**Acceptable error**: $< 10\%$

**If violated**: Data quality issue, missing major ions.

#### 9.2 Residual Patterns

Plot $r_\ell$ vs. $\ell$ (ion index):

- **Random scatter**: Good fit
- **Systematic bias** (all positive or negative): Missing endmember
- **Outliers**: Specific ions poorly fit

#### 9.3 Mineral Fractions

Normalized fractions sum to 1:

$$\sum_{k=1}^K \frac{\hat{s}_k}{\sum_{k'} \hat{s}_{k'}} = 1$$

**Interpretation**: Relative importance of each mineral.

**Caution**: Fractions depend on chosen library; different library → different fractions.

#### 9.4 Sensitivity to $π_G$

Compare inversions with and without $π_G$ weighting:

- Large difference → Attribution matters
- Small difference → Stoichiometry dominates

### 10. Visualization

#### 10.1 Stacked Bar Chart: Mineral Fractions

For each sample $i$, stacked bar with height = 1:

- Segments = mineral fractions
- Color by mineral type (carbonates, sulfates, chlorides, etc.)

#### 10.2 Heatmap: Mineral Plausibility

Heatmap with:

- Rows: Samples
- Columns: Minerals
- Color: Plausibility (binary) or contribution $\hat{s}_{ik}$

#### 10.3 Residual Magnitude

Histogram of $\|r_i\|_D$ across all samples:

- Threshold $\tau_r$ marked
- Fraction above threshold = poor fits

### 11. Hydrogeochemical Constraints & Indices

NeutroHydro integrates standard hydrogeochemical indices to constrain the mineral inversion and provide automatic classification.

#### 11.1 Chloro-Alkaline Indices (CAI)

Used to identify **Ion Exchange** processes.

- **CAI-1**: $[Cl^- - (Na^+ + K^+)] / Cl^-$
- **CAI-2**: $[Cl^- - (Na^+ + K^+)] / (SO_4^{2-} + HCO_3^- + NO_3^-)$

**Constraints**:

- **CAI < 0** (Freshening): Implies $Ca^{2+} \to Na^+$ exchange. The model **bans** `Clay_ReleaseCa`.
- **CAI > 0** (Intrusion): Implies $Na^+ \to Ca^{2+}$ exchange. The model **bans** `Clay_ReleaseNa`.

*Note*: These constraints are disabled if the Cl/Br ratio suggests anthropogenic chloride (which makes CAI unreliable).

#### 11.2 Gibbs Diagram Ratios

Used to classify the dominant hydrogeochemical process.

- **Anion Ratio**: $Cl^- / (Cl^- + HCO_3^-)$
- **Cation Ratio**: $Na^+ / (Na^+ + Ca^{2+})$

**Constraints**:

- **Rock Dominance** (Both ratios < 0.5): The model **penalizes/bans** Evaporite minerals (e.g., Halite, Mirabilite) to prevent overfitting noise as saline deposits in fresh water.

#### 11.3 Simpson's Ratio (Revelle Coefficient) & BEX

The model distinguishes salinity mechanisms using the Simpson Ratio and its variants.

1. **Simpson's Ratio (SR)**:
    - *Also known as*: Revelle Coefficient.
    - *Formula*: $Cl^- / (HCO_3^- + CO_3^{2-})$ (all in **meq/L**).
    - **Severity Classification** (Todd/Simpson):
        - **< 0.5**: Good quality (no/very low influence)
        - **0.5 - 1.3**: Slightly contaminated
        - **1.3 - 2.8**: Moderately contaminated
        - **2.8 - 6.6**: Injuriously contaminated
        - **6.6 - 15.5**: Highly contaminated
        - **> 15.5**: Extremely contaminated (Seawater)

2. **Freshening Ratio (FR)**:
    - *Formula*: $(HCO_3^- + CO_3^{2-}) / Cl^-$
    - **Interpretation**: Large FR indicates freshwater dominance; small FR indicates marine influence.

3. **Base Exchange Index (BEX)**:
    - **Process Indicator**: Infers whether the system trends toward freshening or salinization.
    - *Formula*: $BEX = Na^+ + K^+ + Mg^{2+} - 1.0716 \cdot Cl^-$ (meq/L)
    - **Interpretation**:
        - **BEX > 0**: Freshening trend (freshening/recharge)
        - **BEX < 0**: Salinization trend (intrusion/salinization)
        - **BEX ≈ 0**: No clear base-exchange signal

#### 11.4 WHO Quality Integration

The inversion can be "helped" by the **Quality Assessment** module (`neutrohydro.quality_check`).

- If **Saline Intrusion** is detected (High Na + Cl), the model **overrides** Gibbs constraints to allow Halite/Sylvite.
- If **Gypsum** source is inferred (High Ca + SO4), the model ensures Sulfates are plausible.

This allows the model to be **context-aware**, adapting its constraints based on the specific pollution signature of each sample.

### 12. Usage Example

```python
from neutrohydro.minerals import MineralInverter, convert_to_meq
from neutrohydro.quality_check import add_quality_flags
import numpy as np
import pandas as pd

# 1. Load and Assess Quality
df = pd.read_csv("data.csv")
df_quality = add_quality_flags(df)
quality_flags = df_quality.to_dict('records')

# 2. Convert ion data to meq/L
from neutrohydro.minerals import ION_MASSES, ION_CHARGES
# ... (conversion logic) ...

# 3. Create inverter
inverter = MineralInverter()

# 4. Invert with Quality Constraints
result = inverter.invert(
    c_meq, 
    use_cai_constraints=True,
    use_gibbs_constraints=True,
    quality_flags=quality_flags  # <--- Context-aware override
)

# 5. Access results & Indices
print(f"Simpson Class: {result.indices['Simpson_Class']}")
print(f"BEX Indicator: {result.indices['BEX']}")
print(f"Inferred Sources: {df_quality['Inferred_Sources']}")

# 6. Export to DataFrame
df_res = inverter.results_to_dataframe(result, sample_ids=df['Code'])
```

### 13. Integration with Pipeline

The full pipeline can run mineral inference automatically:

```python
from neutrohydro import NeutroHydroPipeline
from neutrohydro.pipeline import PipelineConfig

config = PipelineConfig(
    run_mineral_inference=True,
    mineral_eta=1.0,
    mineral_tau_s=0.01,
    mineral_tau_r=1.0
)

pipeline = NeutroHydroPipeline(config)
results = pipeline.fit(X, y, c_meq=c_meq)  # Pass ion data in meq/L

# Mineral results available
if results.mineral_result is not None:
    print(results.mineral_result.mineral_fractions)
```

### 13. Extensions

#### 13.1 Bayesian Inversion

Replace NNLS with **Bayesian NNLS**:

- Prior distributions on $s$ (e.g., log-normal)
- Posterior samples via MCMC
- Uncertainty quantification

#### 13.2 Sparse Regularization

Add L1 penalty to encourage **sparse solutions**:

$$\hat{s} = \arg\min_{s \geq 0} \|D(c - As)\|^2 + \lambda \|s\|_1$$

**Effect**: Favors fewer minerals with larger contributions.

#### 13.3 Hierarchical Models

Account for spatial/temporal structure:

$$s_i \sim \text{Distribution}(\theta)$$

where $\theta$ are shared parameters across samples.

## References

1. Parkhurst, D. L., & Appelo, C. A. J. (2013). Description of input and examples for PHREEQC version 3. *US Geological Survey Techniques and Methods*, 6(A43), 497.

2. Güler, C., Thyne, G. D., McCray, J. E., & Turner, A. K. (2002). Evaluation of graphical and multivariate statistical methods for classification of water chemistry data. *Hydrogeology Journal*, 10(4), 455-474.

3. Plummer, L. N., & Back, W. (1980). The mass balance approach: Application to interpreting the chemical evolution of hydrologic systems. *American Journal of Science*, 280(2), 130-142.

---

**End of Mathematical Documentation**

For practical usage examples, see [Quick Start Guide](quickstart.md) and [Examples](examples_basic.md).
