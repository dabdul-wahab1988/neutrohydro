---
title: 'NeutroHydro: A Python Package for Neutrosophic Chemometrics in Groundwater Analysis'
tags:
  - Python
  - groundwater
  - chemometrics
  - neutrosophic logic
  - hydrogeochemistry
  - partial least squares
  - water quality
authors:
  - name: Dickson Abdul-Wahab
    orcid: 0000-0001-7446-5909
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Ebenezer Aquisman Asare
    orcid: 0000-0003-1185-1479
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: University of Ghana, Ghana
   index: 1
 - name: Nuclear Chemistry and Environmental Research Centre, National Nuclear Research Institute (NNRI), Ghana Atomic Energy Commission, Ghana
   index: 2
date: 02 January 2025
bibliography: paper.bib
---

# Summary

Groundwater quality assessment relies on understanding the complex interactions between natural geogenic processes and anthropogenic perturbations. Traditional chemometric approaches often struggle to disentangle these sources due to the inherent uncertainty, ambiguity, and incomplete information in hydrogeochemical datasets. `NeutroHydro` introduces the Neutralization-Displacement Geosystem (NDG) framework with Stoichiometric Inversion, a novel neutrosophic chemometric approach that addresses these challenges by operating in absolute concentration space rather than compositional space. The package provides a mathematically rigorous workflow for decomposing groundwater chemistry into baseline (geogenic) and perturbation (anthropogenic) components, enabling quantitative source attribution and mineral inference.

# Statement of need

Groundwater chemometrics faces several persistent challenges: (1) distinguishing natural baseline chemistry from anthropogenic contamination, (2) handling uncertainty and ambiguity in concentration measurements, (3) performing statistically valid operations in non-compositional space, and (4) linking statistical patterns to physical hydrogeochemical processes through mineral stoichiometry. Existing approaches either use compositional data analysis (CoDa) which operates in log-ratio space and loses direct physical interpretability [@Aitchison1986], or apply standard multivariate methods that fail to properly account for the inherent uncertainty structure in geochemical data [@Reimann2008].

`NeutroHydro` fills this gap by implementing a neutrosophic framework that explicitly represents each ion concentration as a triplet of truth (baseline), indeterminacy (uncertainty), and falsity (perturbation) values [@Smarandache1998]. This representation enables simultaneous modeling of multiple information channels while maintaining mathematical rigor through well-defined Euclidean space operations. The package is designed for hydrogeologists, environmental scientists, and water resource managers who need to:

- Quantitatively separate natural and anthropogenic contributions in groundwater samples
- Assess variable importance with channel-wise decomposition
- Infer plausible mineral sources through stoichiometric constraints
- Handle missing data and measurement uncertainty systematically
- Generate interpretable results aligned with domain knowledge

The framework has been applied to groundwater quality assessment and has demonstrated capability in identifying pollution sources, characterizing baseline water chemistry, and diagnosing hydrogeochemical processes [@Abdul2024].

# Mathematical Framework

## Neutrosophic Data Representation

`NeutroHydro` maps each standardized ion concentration $x_{ij}$ (sample $i$, ion $j$) to a neutrosophic triplet $(T_{ij}, I_{ij}, F_{ij})$ where:

- **Truth** ($T$): Baseline component computed via robust operators (median, low-rank approximation, or robust PCA)
- **Indeterminacy** ($I$): Uncertainty/ambiguity channel quantifying measurement or epistemic uncertainty
- **Falsity** ($F$): Perturbation likelihood derived from standardized residuals

For the Truth channel, the baseline operator $\mathcal{B}$ is applied to the standardized predictor matrix:

$$X_T = \mathcal{B}(X^{std})$$

The residuals are computed as $R = X^{std} - X_T$, and the Falsity channel uses a monotone map:

$$F_{ij} = 1 - \exp\left(-\frac{|R_{ij}|}{\sigma_j}\right)$$

where $\sigma_j$ is the robust scale (median absolute deviation) of residuals for ion $j$.

## Augmented Hilbert Space Regression

The three channels are combined into an augmented predictor matrix:

$$X^{aug} = [X_T \quad \sqrt{\rho_I}X_I \quad \sqrt{\rho_F}X_F] \in \mathbb{R}^{n \times 3p}$$

where $\rho_I$ and $\rho_F$ are channel weights. Elementwise precision weights derived from falsity downweight high-perturbation observations:

$$W_{ij} = \exp(-\lambda_F \cdot F_{ij})$$

Probabilistic Neutrosophic PLS (PNPLS) regression is then performed on the weighted augmented matrix $\tilde{X}^{aug} = W \odot X^{aug}$ using the NIPALS algorithm [@Wold1966; @Mevik2007] to extract latent components that predict a target variable $y$ (e.g., log total dissolved solids).

## Variable Importance Decomposition

A key theoretical contribution is the **L2 decomposition theorem** for Variable Importance in Projection (VIP). For each ion $j$, the aggregate VIP satisfies:

$$VIP_{agg}^2(j) = VIP_T^2(j) + VIP_I^2(j) + VIP_F^2(j)$$

This additive decomposition enables unambiguous attribution of prediction importance to baseline versus perturbation sources. The baseline fraction $\pi_G(j)$ for each ion is:

$$\pi_G(j) = \frac{VIP_T^2(j)}{VIP_T^2(j) + VIP_I^2(j) + VIP_F^2(j)} \in [0,1]$$

Ions with $\pi_G(j) \geq 0.7$ are classified as baseline-dominant (geogenic), while $\pi_G(j) \leq 0.3$ indicates perturbation-dominant (anthropogenic).

## Stoichiometric Mineral Inference

To link statistical patterns to physical processes, `NeutroHydro` implements stoichiometric inversion. Given ion concentrations $c \in \mathbb{R}^m$ in meq/L and a stoichiometric matrix $A \in \mathbb{R}^{m \times K}$ representing $K$ candidate minerals, the weighted non-negative least squares (NNLS) problem is:

$$\hat{s} = \arg\min_{s \geq 0} \|D(c - As)\|_2^2$$

where $D = \text{diag}(\pi_G^{\eta})$ prioritizes baseline ions in the fit. The solution $\hat{s}$ represents plausible mineral contributions, validated through residual norms and contribution thresholds. The package includes a comprehensive mineral library and supports custom mineral definitions.

# Features

`NeutroHydro` provides a complete pipeline implementation:

1. **Preprocessing**: Robust centering and scaling using median and median absolute deviation (MAD) to resist outliers
2. **NDG Encoder**: Multiple baseline operators (global median, hydrofacies-conditioned median, low-rank SVD, robust PCA)
3. **PNPLS Regression**: Augmented space regression with configurable channel weights and precision weighting
4. **NVIP Computation**: Channel-wise variable importance with L2 decomposition
5. **Attribution Analysis**: Ion-level baseline fractions ($\pi_G$) and sample-level attribution ($G_i$)
6. **Mineral Inference**: Weighted NNLS inversion with plausibility assessment, thermodynamic validation via saturation indices, and diagnostic indices (Simpson Ratio, Base Exchange Index, Chloro-Alkaline Indices)
7. **Visualization**: Gibbs diagrams, VIP decomposition plots, mineral fraction charts, and correlation matrices
8. **Quality Assessment**: WHO guideline compliance, redox zonation, and pollution fingerprinting

The package operates in absolute concentration space (mg/L, meq/L) rather than compositional space, preserving physical interpretability and enabling direct application of stoichiometric constraints. All operations occur in well-defined Euclidean spaces with rigorous mathematical guarantees.

# Example Usage

```python
import numpy as np
from neutrohydro import NeutroHydroPipeline

# Prepare data: ion concentrations (mg/L or meq/L) and target
X = ...  # Shape: (n_samples, n_ions)
y = ...  # Target: e.g., log TDS
ion_names = ["Ca2+", "Mg2+", "Na+", "K+", "HCO3-", "Cl-", "SO42-"]

# Run complete pipeline
pipeline = NeutroHydroPipeline()
results = pipeline.fit(X, y, feature_names=ion_names)

# Access results
print(f"Model R²: {results.r2_train:.3f}")
print(f"Baseline fractions (π_G): {results.nsr.pi_G}")
print(f"Baseline-dominant ions: {results.nsr.baseline_labels}")

# Sample-level baseline fraction
print(f"Sample baseline scores (G): {results.sample_attribution.G}")

# Optional: mineral inference (requires meq/L)
if results.mineral_result:
    print(f"Plausible minerals: {results.mineral_result.plausible}")
    print(f"Mineral fractions: {results.mineral_result.mineral_fractions}")
```

# Performance and Testing

The package includes comprehensive unit tests and integration tests covering all modules. Testing includes:

- Preprocessing transformations and inverse transforms
- NDG encoder with multiple baseline types
- PNPLS regression with synthetic and real datasets
- NVIP L2 decomposition verification
- Attribution metrics and classification
- Mineral inversion with stoichiometric constraints
- Thermodynamic validation via PHREEQC integration

The NVIP L2 decomposition theorem is verified numerically to machine precision across diverse datasets, confirming the mathematical correctness of the implementation.

# Acknowledgements

We acknowledge the University of Ghana and the Ghana Atomic Energy Commission for institutional support. We thank the reviewers and the open-source Python scientific computing community, particularly the developers of NumPy [@Harris2020], SciPy [@Virtanen2020], scikit-learn [@Pedregosa2011], pandas [@McKinney2010], and Matplotlib [@Hunter2007], upon which this package is built.

# References
