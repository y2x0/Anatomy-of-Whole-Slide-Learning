# Hybrid MIL Failure Modes And Counterexamples

## 1. Readout invariance does not repair context dependence

Let S be a permutation-invariant mean and C be an order-sensitive recurrence. In
general,

```math
S(C(QH))\neq S(C(H)).
```

The final mean cannot repair the order dependence created before pooling. A
state-space model with mean readout remains order-conditioned.

The same logic applies to a graph: a permutation-invariant readout does not make
two different adjacency matrices equivalent.

## 2. Coarsening and context can disagree

Let P be a child-parent assignment and A a fine graph. If

```math
P^{\mathsf T}A\neq A_P P^{\mathsf T},
```

then graph-then-coarsen and coarsen-then-graph produce different parent states.
A model may therefore be sensitive to whether spatial neighbors are aggregated
before or after region formation.

A toy graph with edges crossing a proposed region boundary exposes the issue:
coarsen-first removes the cross-boundary child identity, while graph-first can
transfer signal across that boundary before compression.

## 3. Double counting

A patch can contribute directly and through several contextual routes. With a
residual graph layer,

```math
\widetilde H=(I+A)HW,
```

a high-degree patch appears in its own state and in many neighbor states. A final
sum can therefore amplify local density even if the task should be density
invariant.

Degree normalization, mean aggregation, or explicit count controls change this
surviving statistic.

## 4. Attention collapse after contextualization

Attention scores may become nearly uniform when contextual states collapse:

```math
\widetilde h_j\approx \widetilde h_k
\quad\Longrightarrow\quad
\alpha_j\approx\alpha_k.
```

Alternatively, a single context state can dominate and produce

```math
\max_j\alpha_j\approx 1.
```

The first hides localization; the second creates brittle sparse selection. Monitor
attention entropy and compare against perturbation effects.

## 5. Geometry mismatch

A coordinate graph can connect patches that are close in pixel space but belong to
different tissue compartments. A feature graph can connect morphologically similar
patches far apart. A hierarchy can group cells by a segmentation artifact. These
geometries encode different equivalence relations and cannot be swapped without
changing the model.

A useful sanity test is to compare outputs under:

`text
- coordinate-preserving feature permutation;
- feature-preserving coordinate permutation;
- graph rewiring at fixed features;
- parent-map rewiring at fixed fine states.
`

Each perturbation isolates a different geometric assumption.

## 6. Hybrid explanation error

For z=R(C(H)), the attention score in R is not the derivative of z with
respect to a pre-context instance. The correct total derivative is

```math
\frac{\partial z}{\partial h_k}
=
\sum_j
\frac{\partial z}{\partial \widetilde h_j}
\frac{\partial \widetilde h_j}{\partial h_k}.
```

Deleting h_k also changes graph edges, transformer attention, state trajectories,
or parent scores if those are recomputed. Any explanation that freezes those
objects is a conditional explanation.

## 7. Computational bottleneck migration

Adding operators can move rather than remove the bottleneck:

```math
O(n^2)
\longrightarrow
O(nm^2+m^2)
\longrightarrow
O(n d^2)
```

may reduce dense pairwise cost but increase memory, routing, or feature
projection cost. Approximation rank, number of graph edges, state dimension, and
number of parent tokens must be reported together.

## 8. Counterexample protocol

A hybrid note should include at least four tests:

`text
1. same features, different graph or parent map;
2. same set, different sequence order;
3. same coarse statistic, different within-parent arrangement;
4. same attention map, different contextual dependency graph.

A model that gives the same answer on every pair may have discarded the claimed
inductive bias. A model that gives different answers must explain which object
changed.
`
