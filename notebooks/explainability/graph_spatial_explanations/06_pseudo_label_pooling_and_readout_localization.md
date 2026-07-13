# Pseudo-Label Pooling and Readout Localization

HEAT pools heterogeneous node states by predicted or externally supplied node
types.

## 1. Type-wise Readout

Let `V_a` be nodes assigned to type `a`. A mean type readout is

```math
z_a
=\frac{1}{|V_a|}
\sum_{v\in V_a}h_v.
```

Concatenating type summaries gives

```math
z=\bigoplus_{a\in\mathcal A}z_a.
```

A pseudo-label error changes both membership and normalization.

## 2. Node Contribution Is Contextual

Even with mean pooling, the derivative with respect to one node is

```math
\frac{\partial z_a}{\partial h_v}
=\frac{1}{|V_a|}I_d
\quad\text{for }v\in V_a,
```

but the total class effect also passes through message-passing states and the
classifier. A node's readout coefficient is not its full graph contribution.

## 3. Pseudo-Label Intervention

For fixed states, changing a node's type from `a` to `b` changes both summaries:

```math
z_a'=
\frac{|V_a|z_a-h_v}{|V_a|-1},
\qquad
z_b'=\frac{|V_b|z_b+h_v}{|V_b|+1}.
```

If the graph context is recomputed, this local algebra is only the first stage
of the intervention.

## 4. Localization Limit

A heatmap generated from `z_a` identifies node membership in a pooled category.
It does not show whether the pseudo-label is correct, whether the node was
important within its type, or whether the type itself is required for the
prediction.

