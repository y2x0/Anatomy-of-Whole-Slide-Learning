# CPLIP MIL-NCE Objective And Gradients

Primary anchor:

- Javed et al. "CPLIP: Zero-Shot Learning for Histopathology with
  Comprehensive Vision-Language Alignment." CVPR 2024.
  https://arxiv.org/abs/2406.05205

## Exact Objective

Let visual bag `i` contain embeddings:

```math
\mathcal{B}_i^v
=
\left\{
u_{i1},\ldots,u_{iM_i}
\right\},
```

and text bag `j` contain:

```math
\mathcal{B}_j^t
=
\left\{
v_{j1},\ldots,v_{jN_j}
\right\}.
```

For temperature `\sigma`, define the bag-pair partition:

```math
A_{ij}
=
\sum_{m=1}^{M_i}
\sum_{n=1}^{N_j}
\exp
\left(
u_{im}^{\top}v_{jn}/\sigma
\right).
```

CPLIP's stated MIL-NCE loss is:

```math
\mathcal{L}_{\mathrm{CPLIP}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{A_{ii}}
{\sum_{j=1}^{B}A_{ij}}.
```

Equivalently, define a bag score:

```math
S_{ij}
=
\log A_{ij}.
```

Then:

```math
\mathcal{L}_{\mathrm{CPLIP}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{\exp(S_{ii})}
{\sum_{j=1}^{B}\exp(S_{ij})}.
```

Thus the method is an image-bag-to-text-bag classifier whose logits are
log-sum-exp pools over all cross-modal instance pairs.

## Within-Bag Responsibilities

For fixed bag pair `(i,j)`, define:

```math
\alpha_{ijmn}
=
\frac{
\exp
\left(
u_{im}^{\top}v_{jn}/\sigma
\right)
}{A_{ij}}.
```

These weights satisfy:

```math
\sum_{m=1}^{M_i}
\sum_{n=1}^{N_j}
\alpha_{ijmn}
=
1.
```

The derivative of the bag score is:

```math
\frac{\partial S_{ij}}
{\partial u_{im}}
=
\frac{1}{\sigma}
\sum_{n=1}^{N_j}
\alpha_{ijmn}v_{jn}.
```

Hence MIL-NCE does not weight all positive pairs equally. High-similarity
pairs receive exponentially greater responsibility.

## Exact Anchor Gradient

Let bag-level softmax probability be:

```math
q_{ij}
=
\frac{A_{ij}}
{\sum_{k=1}^{B}A_{ik}}.
```

Then:

```math
\frac{\partial\mathcal{L}}
{\partial S_{ij}}
=
\frac{1}{B}
\left(
q_{ij}
-
\mathbb{1}\{i=j\}
\right).
```

Combining levels:

```math
\frac{\partial\mathcal{L}}
{\partial u_{im}}
=
\frac{1}{B\sigma}
\sum_{j=1}^{B}
\left(
q_{ij}
-
\mathbb{1}\{i=j\}
\right)
\sum_{n=1}^{N_j}
\alpha_{ijmn}v_{jn}.
```

The gradient has a hierarchical attention interpretation: `q_ij` weights text
bags, while `\alpha_ijmn` weights instance pairs within each bag comparison.

## Temperature Limits

Let pair similarities be `r_{mn}=u_m^{\top}v_n`. As `\sigma` approaches zero:

```math
\sigma
\log
\sum_{m,n}
\exp
\left(
r_{mn}/\sigma
\right)
\longrightarrow
\max_{m,n}r_{mn}.
```

The bag relation becomes existential: one highly compatible pair can explain
the match.

For large `\sigma`, a first-order expansion gives:

```math
A_{ij}
\approx
M_iN_j
+
\frac{1}{\sigma}
\sum_{m,n}
u_{im}^{\top}v_{jn}.
```

Bag cardinality then contributes directly to the score.

## Bag-Size Bias

If every similarity in two candidate text bags is identical at value `r`,
then:

```math
A_{ij}
=
M_iN_j
\exp(r/\sigma),
```

so:

```math
S_{ij}
=
\log M_i
+
\log N_j
+
r/\sigma.
```

Larger text bags receive larger logits even with identical compatibility. A
cardinality-normalized alternative would be:

```math
\widetilde A_{ij}
=
\frac{1}{M_iN_j}
A_{ij},
```

but that is not the stated CPLIP equation.

## Positive Contamination

Partition the positive Cartesian product into valid and invalid subsets:

```math
\mathcal{P}_{ii}
=
\mathcal{P}_{ii}^{+}
\sqcup
\mathcal{P}_{ii}^{-}.
```

The fraction of positive partition mass assigned to invalid pairs is:

```math
\eta_i
=
\frac{
\sum_{(m,n)\in\mathcal{P}_{ii}^{-}}
\exp(r_{mn}/\sigma)
}{
\sum_{m,n}
\exp(r_{mn}/\sigma)
}.
```

Noise is harmless only when its exponential mass is small, not merely when
invalid pairs are numerically rare.

## Asymmetry

The stated equation normalizes text bags for each visual bag. It does not add
the reverse text-bag-to-visual-bag term used by symmetric CLIP. Consequently,
the exact objective need not constrain both conditional retrieval directions
equally.
