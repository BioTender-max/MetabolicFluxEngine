# MetabolicFluxEngine

**Flux Balance Analysis of Metabolic Networks**

A pure-Python pipeline for metabolic flux analysis using linear programming, including FBA, FVA, and gene essentiality prediction.

## Features
- Stoichiometric matrix construction (E. coli core model, 95 reactions)
- FBA (linear programming via scipy.optimize.linprog)
- Flux variability analysis (FVA: min/max per reaction)
- Gene essentiality prediction (single-reaction knockouts)
- GIMME-style expression-integrated flux analysis

## Results
- 95-reaction E. coli core metabolic network, 50 conditions
- FBA optimal biomass: 20.0 mmol/gDW/h
- Mean biomass across conditions: 15.43
- FVA mean range: 157.35
- GIMME-FBA correlation: r=0.974

## Usage
```bash
pip install numpy scipy matplotlib
python metabolic_flux_engine.py
```

## Tags
`flux-analysis` `metabolic-modeling` `fba` `cobra` `stoichiometric-matrix` `systems-biology`
