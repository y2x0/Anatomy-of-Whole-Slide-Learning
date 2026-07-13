# Sparse WSI Concepts and Exact Credit

ProtoMIL turns foundation-model patch embeddings into nonnegative sparse
coordinates and then aggregates them with MIL attention.

## 1. Patch Concepts

```math
h_{ij}
=\mathrm{ReLU}(W_{\mathrm{enc}}x_{ij}+b),
\qquad
\widehat x_{ij}=W_{\mathrm{dec}}h_{ij}.
```

The sparse autoencoder objective is

```math
\sum_{i,j}
\|x_{ij}-\widehat x_{ij}\|_2^2
+\lambda\|h_{ij}\|_1.
```

This encourages a sparse reconstruction dictionary. Concept names are assigned by
inspecting patches with large coordinate activation; they are not supervised by
those names.

## 2. Slide Concept Statistic

With normalized attention weights,

```math
z_i=\sum_{j=1}^{n_i}a_{ij}h_{ij},
\qquad
\sum_ja_{ij}=1.
```

The surviving concept coordinate is a weighted first moment:

```math
z_{ik}=\sum_ja_{ij}h_{ijk}.
```

It mixes prevalence, activation magnitude, and attention routing. It is not a
simple count of concept-positive patches.

## 3. Exact Logit Credit

For a linear head,

```math
F_{ic}=b_c+\sum_kw_{ck}z_{ik},
```

the exact concept credit is

```math
\Gamma_{ikc}=w_{ck}z_{ik}.
```

This is stronger than TCAV's local sensitivity because it algebraically sums to
the current logit. It remains weaker than a causal claim and generally differs
from deletion effect after attention is recomputed.

## 4. Dictionary Failure Surface

A sparse coordinate may be:

```math
\text{polysemantic},
\quad
\text{split across several coordinates},
\quad
\text{unused by the classifier},
\quad
\text{stable only within one encoder or cohort}.
```

Required audits include reconstruction error, activation sparsity, semantic
purity of top patches, cross-seed coordinate matching, classifier utilization,
and fixed-model concept ablation.
