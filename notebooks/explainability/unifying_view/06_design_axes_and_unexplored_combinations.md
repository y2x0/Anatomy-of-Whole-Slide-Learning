# Design Axes and Unexplored Combinations

## 1. Design Tensor

An explainer can be indexed by

```math
E=(O,T,Q,I,V),
```

where:

```text
O = slide object: set, graph, hierarchy, distribution, concept field
T = target: score, hazard, curve, incidence, representation
Q = quantity: weight, gradient, credit, deletion, intervention
I = intervention: none, delete, mask, rewire, decode, retrain
V = validation: stability, faithfulness, plausibility, calibration, shift
```

Most papers occupy one cell or a thin slice of this tensor.

## 2. Underexplored Combinations

The repository's mapped literature leaves especially important combinations to
test rather than declare solved:

```text
distributional slide representation + spatial counterfactual
time-conditioned survival attention + calibrated curve attribution
hierarchical graph + cell-level feasible intervention
prototype mixture + competing-risk explanation
multimodal co-attention + external-domain causal validation
foundation-model concept prior + patient-level survival counterfactual
```

These are design hypotheses, not claims that the literature already supports
them.

## 3. New Method Checklist

Before proposing a new explainer, specify which existing tensor coordinates it
changes. A new heatmap with the same target, quantity, and intervention may be a
new visualization rather than a new mathematical explanation.

