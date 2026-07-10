# Relation Embeddings And Two-Stage Normalization

This note derives WiKG's endpoint-interpolated edge state and separates the graph-construction weights from the later knowledge-aware message weights.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Relation Embedding

For each selected source-target pair, WiKG defines:

```math
r_{vu}
=
\omega_{vu}t_u
+
\left(
1-\omega_{vu}
\right)q_v
\in
\mathbb{R}^{d}.
```

Since selected-neighbor softmax weights satisfy:

```math
0<\omega_{vu}<1
```

for more than one finite-logit neighbor, the relation state lies on the line
segment from target head to source tail:

```math
r_{vu}-q_v
=
\omega_{vu}
\left(
t_u-q_v
\right).
```

Thus the edge state is not an independent relation variable. It is a
score-dependent interpolation of its endpoint roles.

Two consequences follow:

```math
\omega_{vu}\to 1
\quad\Longrightarrow\quad
r_{vu}\to t_u,
```

```math
\omega_{vu}\to 0
\quad\Longrightarrow\quad
r_{vu}\to q_v.
```

The same head-tail logit affects both whether an edge exists and where its edge
embedding lies.

## Two Normalizations

WiKG uses two distinct rowwise softmax operations.

The first creates edge interpolation weights:

```math
\omega_{vu}
=
\mathrm{softmax}_{u\in\mathcal{N}_K(v)}
\left(
s_{vu}
\right).
```

The second, derived in the later context-operator notes, creates message
aggregation weights:

```math
\pi_{vu}
=
\mathrm{softmax}_{u\in\mathcal{N}_K(v)}
\left(
a_{vu}
\right).
```

These are not the same distribution:

```text
omega:
    generated directly from head-tail compatibility

pi:
    generated from a nonlinear function of head, tail, and relation state
```

## Dense Summary

WiKG graph construction is the composition:

```math
\widetilde G
\xrightarrow{W_H,W_T}
(Q,T)
\xrightarrow{QT^{\top}/\sqrt d}
S
\xrightarrow{\text{rowwise top-}K}
A
\xrightarrow{\mathrm{selected\ softmax}}
\Omega
\xrightarrow{\mathrm{endpoint\ interpolation}}
R.
```

The support is sparse and directed, but discovering it is dense and its
discrete identities are only piecewise stable.
