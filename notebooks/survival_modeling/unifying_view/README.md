# Unifying View

This folder ties the survival notebook together.

The main claim:

```text
survival modeling is the choice of a risk object plus a censoring-aware loss.
```

For WSI:

```math
H_i
\xrightarrow{\mathcal{C}}
\widetilde{H}_i
\xrightarrow{\mathcal{R}}
z_i
\xrightarrow{\mathcal{H}}
\mathcal{Y}_i^{\operatorname{surv}}
\xrightarrow{\ell}
\mathcal{L}.
```

## Files

- `01_risk_object_lattice.md`: scalar, curve, density, CIF, and PMF outputs.
- `02_crgs_survival_decomposition.md`: context/readout/geometry/supervision.
- `03_design_axes_and_failure_modes.md`: design table.
- `04_paper_placement_matrix.md`: how representative papers fit.
- `05_minimal_correct_reporting.md`: what every survival WSI note/paper should state.
