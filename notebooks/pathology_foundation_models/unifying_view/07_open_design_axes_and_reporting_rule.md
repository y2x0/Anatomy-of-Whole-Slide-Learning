# Open Design Axes and Reporting Rule

## 1. Design Axes

```text
patch versus slide objective;
appearance versus morphology target;
single stain versus multistain view;
image-only versus language or knowledge alignment;
local versus long-range context;
retrieval versus parametric readout;
teacher transfer versus independent student geometry;
in-domain versus external-cohort validation.
```

The unexplored method space is the Cartesian product of these choices, subject
to compute and data constraints.

## 2. Minimum PFM Report

Every foundation-model paper should report:

```text
pretraining data unit; view construction; positive/negative or mask rule;
teacher or text source; objective; encoder and context geometry; sampling unit;
downstream readout; patient-level split; external-shift test; calibration or
retrieval metric; and failure modes expected from the objective.
```

## 3. Final Placement

```math
\text{PFM claim}
\mapsto
(\mathfrak P,\Phi,\mathcal R,\mathcal H,\text{evaluation},\text{shift}).
```

Without this tuple, “foundation model” is a scale label rather than a
mathematical description.
