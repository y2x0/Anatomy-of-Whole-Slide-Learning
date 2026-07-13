# Design Checklist

Use this checklist when designing or reading an attention method for WSI.

## 1. What Are The Queries?

```text
patch tokens
class tokens
CLS token
PMA seed tokens
pathway tokens
text prompts
memory items
```

Queries define what asks for information.

## 2. What Are The Keys?

```text
patch morphology
coordinates plus morphology
graph node features
modality tokens
prototype states
report/text states
```

Keys define what can be matched.

## 3. What Are The Values?

The value is what survives:

```math
m_u
=
\sum_v a_{uv}r_v.
```

If the value does not contain geometry, prevalence, or burden, attention cannot
recover those after the sum.

## 4. What Is The Support?

```text
all patches
top-k patches
window neighbors
graph edges
cross-modal tokens
memory bank
```

Support determines which interactions are possible.

## 5. What Is The Normalization?

```text
softmax:
    dense probability measure

sparsemax/entmax:
    learned sparse support

masked softmax:
    imposed support plus dense weights inside support

top-k:
    hard support selection
```

## 6. What Statistic Survives?

Examples:

```text
one weighted first moment
one weighted moment per class
one contextual state per patch
one state per seed query
one fused multimodal statistic
```

## 7. What Would Break It?

Check:

```text
rare positives
distributed burden
wrong geometry
query collapse
attention shortcut
normalization by bag size
metric-output mismatch
overinterpreted heat maps
```

## Dense Summary

A new attention method should be described by:

```text
query source
key source
value source
score function
normalization
support
surviving statistic
failure mode
```

If any of these are missing, the method is not mathematically legible yet.

## Minimum Operator Record

At minimum, write the dimensions and the forward map:

```math
Q
\in
\mathbb{R}^{m\times d_k},
\qquad
K
\in
\mathbb{R}^{n\times d_k},
\qquad
R
\in
\mathbb{R}^{n\times d_v},
```

```math
A
=
\sigma_{\mathcal{A}}
\left(
\frac{QK^\top}{\sqrt{d_k}}+B
\right),
\qquad
Z
=
AR.
```

Then state whether `Z` is a token field, a set of query summaries, or the
final slide statistic. A method description that gives only an attention name
but omits this record has not identified its aggregation operator.

## Minimum Sanity Checks

Before claiming a new inductive bias, test:

```text
permutation test:
    does the claimed invariance/equivariance hold without geometry?

support test:
    can a zero mask or top-k rule delete the relevant instance?

bag-size test:
    how does duplicating background change the normalized statistic?

value test:
    can the stated value map encode the claimed surviving statistic?

intervention test:
    does removing the selected patch change the output after renormalization?
```
