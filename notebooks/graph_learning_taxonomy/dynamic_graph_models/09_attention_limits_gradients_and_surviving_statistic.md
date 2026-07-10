# Attention Limits, Gradients, And The Surviving Statistic

This note follows the two nested weight systems through limiting cases and gradient paths, then identifies the exact neighborhood statistic surviving WiKG context.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Two-Stage Weight Dependence

The full source-to-target coefficient is not simply `omega` or `pi`.

The first distribution affects the relation state:

```math
\omega_{vu}
\longrightarrow
r_{vu}.
```

The relation state affects the second logit:

```math
r_{vu}
\longrightarrow
a_{vu}.
```

The second distribution determines the linear message coefficient:

```math
a_{vu}
\longrightarrow
\pi_{vu}
\longrightarrow
m_v.
```

Therefore:

```math
\pi_{vu}
=
\pi_{vu}
\left(
Q,T,\Omega(Q,T)
\right).
```

Calling both distributions "attention" without distinguishing their roles
loses the structure of the operator.

## Limiting Cases

### One Selected Neighbor

If `K=1`, both local softmax distributions collapse:

```math
\omega_{vu}
=
1,
\qquad
\pi_{vu}
=
1.
```

Then:

```math
r_{vu}
=
t_u,
```

```math
m_v
=
t_u.
```

Knowledge-aware attention has no within-neighborhood weighting choice; only
hard neighbor selection remains.

### Uniform Triplet Scores

If:

```math
a_{vu}
=
c_v
```

for all selected sources, then:

```math
\pi_{vu}
=
\frac{1}{K},
```

and:

```math
m_v
=
\frac{1}{K}
\sum_{u\in\mathcal{N}_K(v)}t_u.
```

The context operator becomes a directed learned-neighborhood mean followed by
dual-interaction fusion.

### Saturated Triplet Scores

If one selected source `u_star` dominates:

```math
a_{vu_{\star}}-a_{vu}
\to
+\infty
```

for every other selected `u`, then:

```math
m_v
\to
t_{u_{\star}}.
```

The context operator approaches hard message routing within the already hard
top-k support.

## Gradient Paths

Within a region where the selected neighbor identities are fixed, the slide
loss can update head and tail projections through several paths:

```math
\mathcal{L}
\to
h_v^{+}
\to
m_v
\to
\pi_{vu}
\to
a_{vu}
\to
(q_v,r_{vu},t_u),
```

and:

```math
\mathcal{L}
\to
r_{vu}
\to
\omega_{vu}
\to
s_{vu}
\to
(q_v,t_u).
```

However, unselected nodes receive no gradient through a continuously relaxed
edge-selection probability in the released hard top-k construction. They can
enter the graph only after parameter changes move their logits across a top-k
rank boundary.

## C/R/G/S Placement

```text
\mathcal{G}:
    A and R determine which triplets exist

\mathcal{C}:
    triplet score -> local softmax -> tail aggregation -> dual interaction

\mathcal{R}:
    receives the updated node states but is not part of this local operator

\mathcal{S}:
    slide-level cross-entropy shapes both attention stages through updated nodes
```

## Surviving Node Statistic

Before graph-level readout, each updated node preserves:

```text
target head state
selected-neighbor weighted first moment
coordinatewise interaction between the two
```

Mathematically:

```math
h_v^{+}
\in
\mathcal{F}
\left(
q_v,
\sum_{u\in\mathcal{N}_K(v)}\pi_{vu}t_u,
q_v\odot
\sum_{u\in\mathcal{N}_K(v)}\pi_{vu}t_u
\right).
```

Pairwise information survives only after being compressed into one weighted
neighbor mean per target. Distinct selected neighborhoods that produce the same
`m_v` are indistinguishable to the subsequent dual-interaction update.
