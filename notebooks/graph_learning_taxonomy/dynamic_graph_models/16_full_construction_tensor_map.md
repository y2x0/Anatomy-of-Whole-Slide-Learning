# Full Construction Tensor Map

This note collects the dimension-checked construction path from patch features through head-tail projection, dense compatibility, hard top-k selection, selected normalization, and relation tensors.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Dimensions

For one slide:

```text
n:
    number of foreground tissue patches

d_0:
    patch-encoder dimension; 384 in the reported ViT-Small setup

d:
    hidden graph dimension; 512 in the released configuration

K:
    selected source count per target; 6 in the released training script

C:
    number of slide classes
```

The paper reports one WSI per optimization batch.

## 1. Patch Feature Matrix

Let:

```math
E
=
[e_1;\ldots;e_n]
\in
\mathbb{R}^{n\times d_0}
```

contain fixed or pretrained patch-encoder features.

The released model applies an affine projection and LeakyReLU:

```math
G
=
\phi_{\mathrm{LReLU}}
\left(
EW_{\mathrm{in}}
+
\mathbf{1}_n b_{\mathrm{in}}^{\top}
\right)
\in
\mathbb{R}^{n\times d},
```

with:

```math
W_{\mathrm{in}}
\in
\mathbb{R}^{d_0\times d},
\qquad
b_{\mathrm{in}}
\in
\mathbb{R}^{d}.
```

## 2. Released-Code Slide-Mean Injection

Define:

```math
\overline g
=
\frac{1}{n}
\mathbf{1}_n^{\top}G
\in
\mathbb{R}^{d}.
```

The released code replaces every node row by:

```math
\widetilde G
=
\frac{1}{2}
\left(
G
+
\mathbf{1}_n\overline g
\right)
\in
\mathbb{R}^{n\times d}.
```

This step is executed by the public model but omitted from the paper's
displayed methodology equations.

## 3. Head And Tail Projection

Head matrix:

```math
Q
=
\widetilde GW_H
+
\mathbf{1}_n b_H^{\top}
\in
\mathbb{R}^{n\times d}.
```

Tail matrix:

```math
T
=
\widetilde GW_T
+
\mathbf{1}_n b_T^{\top}
\in
\mathbb{R}^{n\times d}.
```

Row `v` of `Q` is target head `q_v`. Row `u` of `T` is source tail `t_u`.

The projection dimensions are:

```math
W_H,W_T
\in
\mathbb{R}^{d\times d},
\qquad
b_H,b_T
\in
\mathbb{R}^{d}.
```

## 4. Dense Directed Compatibility

The all-pairs score matrix is:

```math
S
=
\frac{1}{\sqrt d}
QT^{\top}
\in
\mathbb{R}^{n\times n}.
```

Its entries are:

```math
S[v,u]
=
s_{vu}
=
\frac{q_vt_u^{\top}}{\sqrt d}.
```

The first index denotes the receiving head; the second denotes the candidate
sending tail.

## 5. Hard Top-K Index Tensor

For every target row `v`, let:

```math
J[v,:]
=
\mathrm{TopKIndices}
\left(
S[v,:],K
\right).
```

Then:

```math
J
\in
\{1,\ldots,n\}^{n\times K}.
```

The selected directed edge set is:

```math
E_{\theta}
=
\left\{
(J[v,k],v):
v=1,\ldots,n,
\ k=1,\ldots,K
\right\}.
```

Define selected logits:

```math
S_K[v,k]
=
S[v,J[v,k]]
\in
\mathbb{R}.
```

Thus:

```math
S_K
\in
\mathbb{R}^{n\times K}.
```

## 6. Selected-Neighbor Similarity Weights

The paper's pseudocode and released implementation normalize selected logits:

```math
\Omega[v,k]
=
\frac{
\exp(S_K[v,k])
}{
\sum_{a=1}^{K}
\exp(S_K[v,a])
}.
```

Therefore:

```math
\Omega
\in
\mathbb{R}^{n\times K},
```

```math
\Omega[v,k]>0,
```

```math
\sum_{k=1}^{K}
\Omega[v,k]
=
1.
```

This is not the raw-dot-product normalization printed in Equation 2 of the
paper. It is the implementation-faithful selected-neighbor softmax.

## 7. Gathered Tail Tensor

Gather selected source tails:

```math
T_K[v,k,:]
=
T[J[v,k],:]
\in
\mathbb{R}^{d}.
```

Then:

```math
T_K
\in
\mathbb{R}^{n\times K\times d}.
```

Broadcast target heads across their selected-source axis:

```math
Q_K[v,k,:]
=
Q[v,:],
```

so:

```math
Q_K
\in
\mathbb{R}^{n\times K\times d}.
```

## 8. Directed Relation Tensor

Broadcast `Omega` over the feature axis. The relation tensor is:

```math
R_K
=
\Omega[:,:,\mathrm{None}]
\odot
T_K
+
\left(
1-
\Omega[:,:,\mathrm{None}]
\right)
\odot
Q_K.
```

Therefore:

```math
R_K
\in
\mathbb{R}^{n\times K\times d}.
```

Entrywise:

```math
R_K[v,k,:]
=
\Omega[v,k]
T_K[v,k,:]
+
\left(
1-\Omega[v,k]
\right)
Q[v,:].
```
