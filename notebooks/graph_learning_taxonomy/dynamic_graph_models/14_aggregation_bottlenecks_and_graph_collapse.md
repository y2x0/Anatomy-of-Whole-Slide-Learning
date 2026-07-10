# Aggregation Bottlenecks And Graph Collapse

This note isolates structural and statistical collapse modes after support selection: fixed degree, self-messages, endpoint interpolation, nested attention, local moment collisions, and global readout collisions.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## 4. Fixed K Imposes Uniform Inbound Capacity

Every target selects exactly `K` source slots when top-k is defined:

```math
\sum_{u=1}^{n}A[v,u]
=
K.
```

This assumes the same relational degree is appropriate for every patch.

A morphologically isolated target is forced to accept `K` sources even if all
scores are poor. A complex tumor-interface target cannot accept more than `K`
direct sources even if many patches are relevant.

The model ranks relations but has no absolute reject threshold:

```math
u\in\mathcal{N}_K(v)
```

means that `u` is among the best available candidates, not that its relation is
strong in an absolute sense.

## 5. Self-Edges Compete With Cross-Patch Edges

The released implementation does not mask diagonal logits:

```math
s_{vv}
=
\frac{q_vt_v^{\top}}{\sqrt d}.
```

If self-compatibility is large:

```math
v
\in
\mathcal{N}_K(v),
```

one selected slot carries the node's own tail state. This can be useful as a
self-message, but it reduces the number of cross-patch sources from `K` to
`K-1` for that target.

If diagonal scores dominate widely, the dynamic graph can approach an
identity-heavy support:

```math
A
\approx
I+A_{\mathrm{residual}}.
```

The graph learner then behaves more like a nodewise transform with limited
relational correction.

## 9. Edge States Are Endpoint Interpolations

WiKG defines:

```math
r_{vu}
=
\omega_{vu}t_u
+
\left(1-\omega_{vu}\right)q_v.
```

Hence:

```math
r_{vu}
\in
\mathrm{conv}
\{q_v,t_u\}.
```

The edge state cannot encode an arbitrary relation vector independent of its
endpoints. If two source-target pairs have matching endpoint mixtures, their
edge states are identical even if an external pathology ontology would assign
different relation types.

When selected-neighbor weights are diffuse:

```math
\omega_{vu}
\approx
\frac{1}{K},
```

and `K` is moderately large, every relation state remains close to its target
head:

```math
r_{vu}
\approx
\left(1-\frac{1}{K}\right)q_v
+
\frac{1}{K}t_u.
```

Neighbor-specific edge differences can then be small relative to the shared
target component.

## 10. Two-Stage Attention Can Collapse Twice

The graph uses selected-neighbor weights:

```math
\omega_{vu}
=
\mathrm{softmax}_{u\in\mathcal{N}_K(v)}(s_{vu})
```

to construct edge states, then knowledge-aware weights:

```math
\pi_{vu}
=
\mathrm{softmax}_{u\in\mathcal{N}_K(v)}(a_{vu})
```

to aggregate tails.

If either distribution has low entropy:

```math
\mathcal{H}(\omega_v)
\approx
0
```

or:

```math
\mathcal{H}(\pi_v)
\approx
0,
```

one neighbor can dominate. The first collapse makes relation construction
nearly tail-valued; the second makes the neighborhood message nearly one tail:

```math
m_v
\approx
t_{u^{\star}}.
```

If both are nearly uniform, the opposite failure occurs: distinct selected
relations are diluted into a neighborhood mean.

## 11. One Weighted Neighbor Mean Is The Local Bottleneck

The entire selected neighborhood becomes:

```math
m_v
=
\sum_{u\in\mathcal{N}_K(v)}
\pi_{vu}t_u.
```

Suppose two different neighborhoods satisfy:

```math
\sum_{u\in\mathcal{N}_K(v)}
\pi_{vu}t_u
=
\sum_{a\in\mathcal{N}'_K(v)}
\pi'_{va}t'_a.
```

They produce the same `m_v` and are indistinguishable to dual-interaction
fusion for fixed `q_v`.

The local readout preserves a weighted first moment, not the full distribution
of selected source states or all pairwise source-source interactions.

## 12. Global Readout Can Erase The Dynamic Graph

Under the released training script, graph-level mean pooling gives:

```math
z
=
\frac{1}{n}
\sum_{v=1}^{n}h_v^{+}.
```

The topology can alter each updated node, but all updated nodes are then
compressed to one first moment. Two slides with different dynamic graphs can
produce the same pooled statistic:

```math
A_1
\ne
A_2,
```

```math
\frac{1}{n_1}
\sum_v h_{1v}^{+}
=
\frac{1}{n_2}
\sum_v h_{2v}^{+}.
```

Then the same LayerNorm and classifier produce the same prediction. A richer
graph does not guarantee a richer surviving slide statistic.
