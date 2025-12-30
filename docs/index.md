# NeutroHydro Documentation

## Neutrosophic Chemometrics for Groundwater Analysis

## Table of Contents

### Getting Started

- [Quick Start Guide](quickstart.md)
- [Installation](installation.md)

### Mathematical Foundations

- [Mathematical Framework Overview](mathematical_framework.md)
- [Preprocessing & Robust Scaling](preprocessing.md)
- [NDG Encoder: Neutrosophic Triplets](encoder.md)
- [PNPLS: Probabilistic Neutrosophic PLS](model.md)
- [NVIP: Variable Importance Decomposition](nvip.md)
- [Attribution: NSR and Baseline Fractions](attribution.md)
- [Mineral Stoichiometric Inversion](minerals.md)
- [Water Quality Assessment](quality_check.md)
- [Model Limitations & Validity](limitations.md)
- [Hydrogeochemical Processes](hydrogeochemical_processes.md): Mixing, Exchange, Redox
- [Mathematical Critique](mathematical_critique.md): Rigorous review of potential issues
- [Final Critical Review](final_critical_review.md): "Red Team" analysis of validity

### API Reference

- [Pipeline API](api_pipeline.md)
- [Core Modules API](api_modules.md)

### Examples & Tutorials

- [Basic Usage Example](examples_basic.md)
- [Advanced Workflows](examples_advanced.md)
- [Interpreting Results](interpreting_results.md)

## Overview

NeutroHydro implements a mathematically well-posed workflow for groundwater chemometrics in **absolute concentration space** (non-compositional):

```text
Raw Ion Data
     ↓
Preprocessing (Robust centering/scaling)
     ↓
NDG Encoder (T, I, F triplets)
     ↓
PNPLS Regression (Augmented Hilbert space)
     ↓
NVIP (Channel-wise variable importance)
     ↓
NSR/π_G (Baseline vs perturbation attribution)
     ↓
Mineral Inference (Stoichiometric inversion)
```

## Core Mathematical Innovations

### 1. Neutrosophic Data Representation

Maps each ion concentration to a triplet **(T, I, F)**:

- **T (Truth)**: Baseline/reference component
- **I (Indeterminacy)**: Uncertainty/ambiguity
- **F (Falsity)**: Perturbation likelihood

### 2. L2-Additive VIP Decomposition

**Theorem**: Variable importance decomposes additively across channels:

$$VIP_{agg}^2(j) = VIP_T^2(j) + VIP_I^2(j) + VIP_F^2(j)$$

This allows **unambiguous attribution** of prediction importance to baseline vs. perturbation sources.

### 3. Non-Compositional Framework

Unlike compositional data analysis (CoDa), NeutroHydro operates in **absolute concentration space**, preserving:

- Physical interpretability
- Additive mixing models
- Direct stoichiometric constraints

### 4. Hybrid Geochemical-Statistical Engine

Combines rigorous mathematical optimization with expert hydrogeochemical heuristics:

- **Context-Aware Inversion**: Uses **WHO Quality Flags** and **Gibbs Diagrams** to dynamically constrain the mineral solver.
- **Thermodynamic Validation**: Integrated **PHREEQC** engine for Saturation Index (SI) calculation and redox-aware speciation (Eh) to ensure identified mineral assemblages are physically realistic.
- **Redox Detection**: Explicitly solves for mass loss (e.g., Denitrification) using negative stoichiometry.
- **Advanced Indices**: Integrated **Simpson's Ratio** (Revelle Coefficient) and **Base Exchange Index (BEX)** for precise salinity and freshening diagnosis.

## Quick Navigation

**For Users:**

- New to NeutroHydro? → [Quick Start Guide](quickstart.md)
- Need to understand results? → [Interpreting Results](interpreting_results.md)
- Looking for examples? → [Basic Examples](examples_basic.md)

**For Researchers:**

- Mathematical theory? → [Mathematical Framework](mathematical_framework.md)
- Specific module details? → See individual module docs
- Implementation details? → [API Reference](api_modules.md)

**For Developers:**

- Contributing? → See `CONTRIBUTING.md` in repo root
- Testing? → See `tests/` directory

## Citation

If you use NeutroHydro in your research, please cite:

```bibtex
@software{neutrohydro,
  title = {NeutroHydro: Neutrosophic Chemometrics for Groundwater Analysis},
  year = {2024},
     url = {https://github.com/dabdul-wahab1988/neutrohydro}
}
```

## License

MIT License - see LICENSE file for details.
