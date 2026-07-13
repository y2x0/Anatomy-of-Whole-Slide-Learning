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

## Search Over The Design Tensor

The tensor is a product of choices, not a claim that every combination is valid.
For a proposed explainer e, define its placement vector

```math
\zeta(e)
=
\left(
\mathcal O_e,
q_e,
Q_e,
\mathcal I_e,
\mathcal V_e
\right).
```

Two explainers are mathematically distinct only if they differ in at least one
coordinate or induce a different functional on the same coordinate. This gives a
blunt novelty check:

```math
\zeta(e_1)=\zeta(e_2)
\quad\text{and}\quad
E_{e_1}=E_{e_2}
\quad\Longrightarrow\quad
\text{the proposed method is a reparameterization or visualization variant}.
```

An unexplored combination should also specify the missing validity condition. For
example, a time-conditioned graph explanation needs a graph intervention that
preserves the event-time semantics, not just a separate heatmap for every time
bin.
