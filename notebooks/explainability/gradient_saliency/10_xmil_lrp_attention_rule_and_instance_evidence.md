# xMIL-LRP Attention Rule And Instance Evidence

Primary anchor:

- Schramowski et al. "xMIL: Insightful Explanations for Multiple Instance
  Learning in Histopathology." 2024.
  https://arxiv.org/abs/2406.04280

## xMIL Evidence Function

xMIL distinguishes aggregation from evidence. For bag:

```math
X
=
\left\{
x_1,\ldots,x_K
\right\},
```

the learned model estimates an aggregation map, while explanation estimates
signed, context-dependent evidence:

```math
\widehat\epsilon_k
\approx
\mathcal{E}
\left(
X,Y,x_k
\right).
```

## Attention Output

For token embeddings `z_k` and attention scores `p_kj`:

```math
y_j
=
\sum_k
p_{kj}z_k.
```

xMIL uses the AH-rule, treating attention scores as constants during relevance
propagation.

For feature dimension `d`:

```math
R(z_{kd})
=
\sum_j
\frac{
z_{kd}p_{kj}
}{
\sum_i
z_{id}p_{ij}
}
R(y_{jd}).
```

## Conservation Of AH Rule

Summing over source tokens:

```math
\sum_k
R(z_{kd})
=
\sum_j
\left[
\frac{
\sum_kz_{kd}p_{kj}
}{
\sum_iz_{id}p_{ij}
}
\right]
R(y_{jd})
=
\sum_j
R(y_{jd}).
```

This holds where denominators are well-defined.

## Fixed-Attention Boundary

The AH-rule does not propagate relevance through how attention scores change
with inputs:

```math
\frac{\partial p_{kj}}
{\partial z_i}
```

is not part of the rule. It decomposes the realized attention-weighted forward
map using fixed routing coefficients.

## Instance Relevance

At input instance `x_k` with feature relevance vector:

```math
r_k
=
\left[
r_{k1},\ldots,r_{kD}
\right],
```

xMIL defines instance evidence estimate:

```math
\widehat\epsilon_k
=
\sum_{d=1}^{D}
r_{kd}.
```

The bag-level conservation property is:

```math
\sum_k
\widehat\epsilon_k
=
\sum_{k,d}
r_{kd}
=
y,
```

for selected model output `y`, subject to propagation-rule conservation.

## Positive, Negative, Neutral

Unlike nonnegative attention:

```math
\widehat\epsilon_k>0
```

supports the explained target,

```math
\widehat\epsilon_k<0
```

refutes it, and near-zero relevance is neutral.

## Context Sensitivity

Because relevance is propagated through the full MIL model, the same patch
embedding can receive different evidence in different bags. This is necessary
for interaction tasks such as adjacent-pair logic.
