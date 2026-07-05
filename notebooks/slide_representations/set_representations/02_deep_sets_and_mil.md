# Deep Sets And MIL

The Deep Sets form states that many permutation-invariant functions can be
written as:

```math
f(\{h_1,\ldots,h_n\})
=
\rho\left(
\sum_{j=1}^{n}\phi(h_j)
\right).
```

For WSI:

```math
z_i
=
\sum_{j=1}^{n_i}\phi(h_{ij}),
\qquad
\widehat{y}_i=\rho(z_i).
```

This separates the model into:

```text
instance transform
sum aggregation
bag-level head
```

## Mean Pooling As Deep Set

Mean pooling is:

```math
z_i=\frac{1}{n_i}\sum_jh_{ij}.
```

This is a Deep Set with:

```math
\phi(h)=h,
\qquad
\rho=\text{task head},
```

plus normalization by $n_i$.

It preserves only the first moment of the patch embedding distribution.

## Learned Instance Transform

A richer set readout is:

```math
z_i=\frac{1}{n_i}\sum_j\phi_\theta(h_{ij}).
```

The transform can map patches into a space where the mean is more informative.

For example, if $\phi_\theta$ maps tumor-like patches to a high-risk coordinate,
then the mean estimates prevalence of that morphology.

## Max MIL

Classic positive-instance MIL can be written:

```math
\widehat{y}_i
=
\max_j g_\theta(h_{ij}).
```

The set statistic is:

```text
largest instance score
```

This is useful for sparse positives but unstable when patch-level evidence is
noisy.

## Attention MIL

Attention MIL defines:

```math
a_{ij}
=
\frac{\exp(s_\theta(h_{ij}))}
{\sum_{\ell=1}^{n_i}\exp(s_\theta(h_{i\ell}))},
\qquad
z_i=\sum_ja_{ij}v_\theta(h_{ij}).
```

This is still permutation invariant if scores are computed independently over
instances and normalized over the set.

The surviving statistic is:

```math
z_i
=
\mathbb{E}_{j\sim a_i}[v_\theta(h_{ij})].
```

## Set Transformer

Set Transformer introduces attention between set elements while preserving
permutation equivariance before invariant pooling.

If:

```math
\widetilde{H}_i
=
\operatorname{SetAttention}(H_i),
```

then:

```math
\widetilde{H}_{i,\pi}
=
P_\pi\widetilde{H}_i.
```

An invariant readout gives:

```math
z_i=\operatorname{Pool}(\widetilde{H}_i).
```

This allows pairwise or higher-order patch interactions without explicit
geometry.

## Dense Summary

```math
f(H)
=
\rho\left(\sum_{h\in H}\phi(h)\right)
```

is the canonical set-learning template. MIL methods differ by what statistic
replaces or modifies the sum:

```text
mean:
    first moment

max:
    extreme instance score

attention:
    learned weighted first moment

set transformer:
    interaction-aware set states plus invariant readout
```
