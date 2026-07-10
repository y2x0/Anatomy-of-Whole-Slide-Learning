# Directed Score Geometry

This note derives the dense head-tail score matrix, the source of directedness, and the learned bilinear geometry underlying WiKG adjacency.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Score Matrix

For one slide, collect head and tail rows into:

```math
Q
\in
\mathbb{R}^{n\times d},
\qquad
T
\in
\mathbb{R}^{n\times d}.
```

The released implementation computes:

```math
S
=
\frac{1}{\sqrt d}QT^{\top}
\in
\mathbb{R}^{n\times n}.
```

Entry `S[v,u]` scores source tail `u` for target head `v`:

```math
s_{vu}
=
\frac{q_vt_u^{\top}}{\sqrt d}.
```

Scaling controls the variance of dot-product logits. If coordinates of `q_v`
and `t_u` are approximately centered, independent, and have unit variance,
then:

```math
\mathrm{Var}
\left[
q_vt_u^{\top}
\right]
\approx
d,
```

whereas:

```math
\mathrm{Var}
\left[
\frac{q_vt_u^{\top}}{\sqrt d}
\right]
\approx
1.
```

## Directedness Is Not Just Sparse Attention

The reverse logit is:

```math
s_{uv}
=
\frac{q_ut_v^{\top}}{\sqrt d}.
```

For symmetry to hold for all node pairs, one sufficient condition is:

```math
W_H W_T^{\top}
=
W_T W_H^{\top}
```

with compatible treatment of biases. Independent head and tail projections do
not enforce this condition. Even if the score matrix happened to be symmetric,
rowwise top-k selection can still produce asymmetric support at finite `K`.

Therefore the following directed configuration is allowed:

```math
A[v,u]
=
1,
\qquad
A[u,v]
=
0.
```

## Geometry Induced By Head-Tail Compatibility

The learned score can be written as a bilinear form in the pre-projection node
states. Ignoring biases:

```math
s_{vu}
=
\frac{\widetilde g_v
W_H W_T^{\top}
\widetilde g_u^{\top}}
{\sqrt d}.
```

Let:

```math
M_{HT}
=
W_H W_T^{\top}.
```

Then WiKG's topology is a directed top-k graph under the learned bilinear
compatibility:

```math
s_{vu}
=
\frac{
\widetilde g_vM_{HT}\widetilde g_u^{\top}
}{\sqrt d}.
```

If `M_HT` is not symmetric positive semidefinite, the score is not even a
symmetric inner-product kernel. Even when `M_HT` is positive semidefinite, the
bilinear score is a similarity rather than a distance and does not itself
satisfy metric axioms. Calling it a learned relation is more precise than
calling it learned physical distance.

## C/R/G/S Placement

```text
\mathcal{G}:
    support A_theta and relation states R_theta from head-tail logits

\mathcal{C}:
    not yet applied; this note constructs its domain of interaction

\mathcal{R}:
    not part of top-k construction

\mathcal{S}:
    slide loss differentiates through selected weights but not through a
    smooth relaxation of the discrete neighbor identities
```
