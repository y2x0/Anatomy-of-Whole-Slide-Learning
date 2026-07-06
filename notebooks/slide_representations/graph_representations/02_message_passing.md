# Message Passing

Graph neural networks update node states by aggregating neighbor information.

Initialize:

```math
h_v^{(0)}=h_v.
```

Layer $\ell$:

```math
m_v^{(\ell)}
=
\operatorname*{AGG}_{u\in\mathcal{N}(v)}
\psi_\ell(h_v^{(\ell)},h_u^{(\ell)},e_{uv}),
```

```math
h_v^{(\ell+1)}
=
\phi_\ell(h_v^{(\ell)},m_v^{(\ell)}).
```

After $L$ layers:

```math
\widetilde{H}=\{h_v^{(L)}\}_{v\in V}.
```

## GCN Form

A simple graph convolution:

```math
H^{(\ell+1)}
=
\sigma(\widetilde{D}^{-1/2}\widetilde{A}\widetilde{D}^{-1/2}
H^{(\ell)}W_\ell).
```

where:

```math
\widetilde{A}=A+I.
```

This mixes each node with normalized neighbors.

## Graph Attention

Attention coefficient:

```math
\alpha_{uv}
=
\operatorname*{softmax}_{u\in\mathcal{N}(v)}
a_\theta(h_v,h_u,e_{uv}).
```

Update:

```math
h_v'
=
\sigma
\left(
\sum_{u\in\mathcal{N}(v)}
\alpha_{uv}W h_u
\right).
```

Graph attention learns which neighbors matter.

## Permutation Equivariance

If node labels are permuted:

```math
H'=PH,
\qquad
A'=PAP^\top,
```

then message passing satisfies:

```math
\operatorname{GNN}(H',A')
=
P\operatorname{GNN}(H,A),
```

assuming aggregation is permutation invariant over neighbors.

This is the correct symmetry for graph node states.

## Receptive Field

After $L$ message-passing layers, $h_v^{(L)}$ depends on nodes within $L$ hops:

```math
h_v^{(L)}
=
F_\theta
\left(
\{h_u^{(0)}:d_G(u,v)\le L\}
\right).
```

Thus graph depth controls spatial/contextual range.

## Expressivity Limit

Standard message-passing GNNs update a node from the multiset of neighbor
states:

```math
h_v^{(\ell+1)}
=
\Phi_\ell
\left(
h_v^{(\ell)},
\{\!\{h_u^{(\ell)}:u\in\mathcal{N}(v)\}\!\}
\right).
```

Because the neighborhood enters as a multiset, many message-passing GNNs are no
more discriminative than the 1-dimensional Weisfeiler-Lehman color refinement
test. If two graphs produce identical rooted neighborhood multisets at every
iteration, the GNN may assign them the same graph representation even when their
biological meaning differs.

For WSI graphs this matters because coordinate kNN graphs can have very similar
local degree patterns across slides:

```text
different tissue organization
same local graph neighborhoods
```

The model can still use node features, edge features, coordinates, hierarchy, or
global readout to separate slides, but vanilla message passing alone does not
give arbitrary graph expressivity.

## Bottlenecks And Oversquashing

If many distant nodes must influence $v$ through a small cut $B$, then their
information is compressed into bounded-dimensional messages:

```math
\{h_u:u\in U\}
\longrightarrow
\{m_b:b\in B\}
\longrightarrow
h_v^{(L)}.
```

When $|U|$ is large, $|B|$ is small, and message dimension is fixed, long-range
signals can be squashed before reaching the readout. Increasing depth expands
the receptive field but does not automatically increase the capacity of narrow
graph cuts.

## Dense Summary

```math
\boxed{
h_v^{(\ell+1)}
=
\phi_\ell
\left(
h_v^{(\ell)},
\operatorname*{AGG}_{u\in\mathcal{N}(v)}
\psi_\ell(h_v^{(\ell)},h_u^{(\ell)},e_{uv})
\right)
}
```

Message passing turns patch or cell features into context-aware tissue-unit
features.
