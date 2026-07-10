# Full Context, Readout, And Prediction Map

This note continues the tensor map from relation states through paper or code triplet scores, knowledge attention, dual interaction, dropout, graph readout, LayerNorm, logits, and cross-entropy.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## 9. Head-Relation Gate Tensor

The nonlinear gate is:

```math
B_K
=
\tanh
\left(
Q_K+R_K
\right)
\in
\mathbb{R}^{n\times K\times d}.
```

This tensor has one gated relation vector for each selected directed edge.

## 10A. Paper Triplet Score

The paper's Equation 6 gives:

```math
A_K^{\mathrm{paper}}[v,k]
=
T_K[v,k,:]
B_K[v,k,:]^{\top}.
```

Equivalently:

```math
A_K^{\mathrm{paper}}
=
\mathrm{ReduceSum}_{d}
\left(
T_K\odot B_K
\right)
\in
\mathbb{R}^{n\times K}.
```

## 10B. Released-Code Triplet Score

The released einsum assigns different labels to the two feature axes. It
therefore computes:

```math
A_K^{\mathrm{code}}[v,k]
=
\left(
\sum_{r=1}^{d}T_K[v,k,r]
\right)
\left(
\sum_{s=1}^{d}B_K[v,k,s]
\right).
```

In tensor form:

```math
A_K^{\mathrm{code}}
=
\mathrm{ReduceSum}_{d}(T_K)
\odot
\mathrm{ReduceSum}_{d}(B_K)
\in
\mathbb{R}^{n\times K}.
```

Choose one of:

```math
A_K
\in
\left\{
A_K^{\mathrm{paper}},
A_K^{\mathrm{code}}
\right\}
```

depending on whether the target is the published equation or exact public-code
reproduction.

## 11. Knowledge-Aware Attention

Normalize triplet scores over the selected-source axis:

```math
\Pi[v,k]
=
\frac{
\exp(A_K[v,k])
}{
\sum_{a=1}^{K}
\exp(A_K[v,a])
}.
```

Thus:

```math
\Pi
\in
\mathbb{R}^{n\times K},
```

```math
\sum_{k=1}^{K}
\Pi[v,k]
=
1.
```

## 12. Neighborhood Message Matrix

Aggregate selected tails:

```math
M[v,:]
=
\sum_{k=1}^{K}
\Pi[v,k]
T_K[v,k,:].
```

Hence:

```math
M
\in
\mathbb{R}^{n\times d}.
```

Every row is a selected-neighbor weighted first moment.

## 13. Dual-Interaction Update

Additive branch:

```math
H_{\mathrm{sum}}
=
\phi_{\mathrm{LReLU}}
\left(
(Q+M)W_1
+
\mathbf{1}_n b_1^{\top}
\right)
\in
\mathbb{R}^{n\times d}.
```

Multiplicative branch:

```math
H_{\mathrm{prod}}
=
\phi_{\mathrm{LReLU}}
\left(
(Q\odot M)W_2
+
\mathbf{1}_n b_2^{\top}
\right)
\in
\mathbb{R}^{n\times d}.
```

Updated nodes:

```math
H^{+}
=
H_{\mathrm{sum}}
+
H_{\mathrm{prod}}
\in
\mathbb{R}^{n\times d}.
```

Here:

```math
W_1,W_2
\in
\mathbb{R}^{d\times d},
\qquad
b_1,b_2
\in
\mathbb{R}^{d}.
```

## 14. Dropout

For released dropout rate:

```math
p
=
0.3,
```

the training-time tensor is:

```math
\overline H
=
\frac{D}{1-p}
\odot
H^{+},
```

where:

```math
D[v,r]
\sim
\mathrm{Bernoulli}(1-p).
```

At evaluation:

```math
\overline H
=
H^{+}.
```

## 15. Graph Readout

The released training script chooses mean pooling:

```math
z
=
\frac{1}{n}
\mathbf{1}_n^{\top}\overline H
\in
\mathbb{R}^{d}.
```

The model class also supports coordinatewise max:

```math
z_r
=
\max_v\overline H[v,r],
```

or global attention:

```math
\alpha_v
=
\frac{
\exp
\left(
\phi_{\mathrm{LReLU}}
(\overline h_vW_A+b_A)
w_2
+
b_2
\right)
}{
\sum_{u=1}^{n}
\exp
\left(
\phi_{\mathrm{LReLU}}
(\overline h_uW_A+b_A)
w_2
+
b_2
\right)
},
```

```math
z
=
\sum_{v=1}^{n}
\alpha_v\overline h_v.
```

For attention readout:

```math
W_A
\in
\mathbb{R}^{d\times(d/2)},
\qquad
w_2
\in
\mathbb{R}^{(d/2)\times 1}.
```

## 16. LayerNorm And Class Logits

Normalize the pooled vector:

```math
\widetilde z
=
\mathrm{LN}_{\gamma,\beta}(z)
\in
\mathbb{R}^{d}.
```

Class logits are:

```math
\ell
=
\widetilde zW_C+b_C
\in
\mathbb{R}^{C}.
```

The classifier dimensions are:

```math
W_C
\in
\mathbb{R}^{d\times C},
\qquad
b_C
\in
\mathbb{R}^{C}.
```

The public model returns `ell`.

## 17. Probabilities And Loss

Probabilities are:

```math
p_c
=
\frac{\exp(\ell_c)}
{\sum_{a=1}^{C}\exp(\ell_a)}.
```

For observed class `c_star`:

```math
\mathcal{L}_{\mathrm{CE}}
=
-\ell_{c_{\star}}
+
\log
\sum_{a=1}^{C}
\exp(\ell_a).
```
