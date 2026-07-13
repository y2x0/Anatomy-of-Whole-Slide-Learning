# Message Path Attribution and Chain Rules

Node heatmaps do not reveal which messages produced the node state.

## 1. One Message-Passing Layer

For target node `v`, a generic attention update is

```math
m_v^{(\ell)}
=\sum_{u\in\mathcal N(v)}
a_{vu}^{(\ell)}
\psi_\ell(h_v^{(\ell)},h_u^{(\ell)},e_{uv}),
```

```math
h_v^{(\ell+1)}
=\phi_\ell(h_v^{(\ell)},m_v^{(\ell)}).
```

The edge attention `a_vu` is only one factor in the message.

## 2. Local Differential Path

For a scalar output `F`, the differential contribution through one message is
governed by

```math
\frac{\partial F}{\partial a_{vu}^{(\ell)}}
=\frac{\partial F}{\partial h_v^{(\ell+1)}}
\frac{\partial h_v^{(\ell+1)}}{\partial m_v^{(\ell)}}
\psi_\ell(h_v^{(\ell)},h_u^{(\ell)},e_{uv}).
```

High attention with a small message can matter less than low attention with a
large signed message.

## 3. Multi-layer Paths

For an input node `u` and output `F`, every graph path contributes through a
product of Jacobians:

```math
\frac{\partial F}{\partial h_u^{(0)}}
=\sum_{\pi:u\leadsto r}
J_{F\leftarrow r}^{(\pi)}
\prod_{(a\to b)\in\pi}
J_{b\leftarrow a}^{(\pi)}.
```

The sum is over computational paths, not merely spatial shortest paths.

## 4. Nonlinear and Recomputed Effects

Gradients are local and can be saturated. Node deletion includes support,
normalization, and pooling changes. Neither can be identified with edge
attention alone.

## 5. Practical Explanation Stack

For a graph prediction, report at least:

```text
node removal effect; edge or message attention; signed gradient or path score;
and the graph construction used by the intervention.
```

