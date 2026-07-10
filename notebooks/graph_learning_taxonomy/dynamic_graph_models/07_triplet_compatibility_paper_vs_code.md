# Triplet Compatibility: Paper Versus Code

This note defines the edge-conditioned triplet input and proves that the paper's dot-product kernel differs from the rank-one coordinate-sum kernel executed by the released einsum.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Inputs To The Context Operator

For target node `v` and selected source node `u`, the operator receives:

```math
q_v
\in
\mathbb{R}^{d},
\qquad
t_u
\in
\mathbb{R}^{d},
```

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

The selected support is:

```math
u
\in
\mathcal{N}_K(v).
```

The context map is therefore edge-conditioned:

```math
\mathcal{C}_{\theta}^{\mathrm{WiKG}}:
\left(
Q,T,R,A
\right)
\longmapsto
H^{+}.
```

## Paper Triplet Compatibility

WiKG computes the scalar compatibility:

```math
a_{vu}
=
t_u
\left[
\tanh
\left(
q_v+r_{vu}
\right)
\right]^{\top}.
```

Dimensionally:

```math
q_v+r_{vu}
\in
\mathbb{R}^{d},
```

```math
\tanh(q_v+r_{vu})
\in
\mathbb{R}^{d},
```

```math
t_u
\left[
\tanh(q_v+r_{vu})
\right]^{\top}
\in
\mathbb{R}.
```

Substituting the edge state gives:

```math
a_{vu}
=
t_u
\left[
\tanh
\left(
\left(
2-\omega_{vu}
\right)q_v
+
\omega_{vu}t_u
\right)
\right]^{\top}.
```

Thus the same selected-neighbor weight `omega` that constructed the edge also
controls the nonlinear mixture scored by knowledge-aware attention.

This is the operator printed in Equation 6 of the paper. It is also the
dimensionally natural interpretation of a tail vector scoring a gated
head-relation vector.

## Released-Code Einsum Discrepancy

The released forward pass uses:

```text
torch.einsum('...l,...m->...', tail, gate)
```

The two feature axes have different labels, so they are summed independently.
The executed scalar is therefore:

```math
a_{vu}^{\mathrm{code}}
=
\left(
\sum_{r=1}^{d}[t_u]_r
\right)
\left(
\sum_{s=1}^{d}
[\tanh(q_v+r_{vu})]_s
\right).
```

Equivalently:

```math
a_{vu}^{\mathrm{code}}
=
t_u
\mathbf{1}_d\mathbf{1}_d^{\top}
\left[
\tanh(q_v+r_{vu})
\right]^{\top}.
```

The paper equation would require a shared feature index:

```text
torch.einsum('...r,...r->...', tail, gate)
```

which gives:

```math
a_{vu}^{\mathrm{paper}}
=
t_u
\left[
\tanh(q_v+r_{vu})
\right]^{\top}.
```

These are different attention kernels for dimensions greater than one. The
paper kernel pairs matching feature coordinates. The released-code kernel is a
rank-one bilinear form that multiplies two coordinate sums.

The remainder of this note uses `a_vu` for either explicitly chosen kernel.
When interpreting the method as published mathematics, use the paper kernel.
When reproducing the public code exactly, use the code kernel.

## Source-Code Caveat

The released model initializes three modules named `gate_U`, `gate_V`, and
`gate_W`, but the forward pass does not use them. Neither the paper triplet
score nor the executed einsum uses those parameters after head and tail
projection.

Paper equation:

```math
t_u
\left[
\tanh(q_v+r_{vu})
\right]^{\top}.
```

Released-code equation:

```math
\left(
t_u\mathbf{1}_d
\right)
\left(
\mathbf{1}_d^{\top}
\left[
\tanh(q_v+r_{vu})
\right]^{\top}
\right).
```

These notes do not insert the unused gate modules into either equation.
