# Deep Sets And MIL

The Deep Sets template writes a permutation-invariant set function as:

```math
f(\{h_1,\ldots,h_n\})
=
\rho\left(
\sum_{j=1}^{n}\phi(h_j)
\right).
```

The theorem should be read with conditions. For countable domains, invariant
functions admit this kind of sum-decomposition with a suitable feature map. For
continuous set functions on compact domains, neural implementations are best
understood as approximation families. The important modeling lesson is:

```text
per-instance transform, invariant sum, bag-level head
```

not that every finite neural architecture exactly realizes every invariant
functional.

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

This is an extreme-value statistic, not a first moment. It is useful for sparse
positives but unstable when patch-level evidence is noisy.

Max pooling is still permutation invariant and can be approximated by smooth
sum-style statistics. For scalar scores $s_j=g_\theta(h_j)$:

```math
\max_j s_j
=
\lim_{\beta\to\infty}
\frac{1}{\beta}
\log
\left(
\sum_j\exp(\beta s_j)
\right).
```

Thus the right distinction is not:

```text
Deep Sets versus max
```

but:

```text
moment-like prevalence statistic versus extreme-evidence statistic
```

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

The gated attention variant scores instances with:

```math
s_\theta(h)
=
w^\top
\left(
\tanh(Vh)
\odot
\operatorname{sigmoid}(Uh)
\right).
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

Full set attention can be viewed as message passing on the complete graph over
instances:

```math
\widetilde{h}_j
=
\sum_k
\operatorname*{softmax}_{k}
\left(
\frac{q_j^\top k_k}{\sqrt{d}}
\right)
v_k.
```

Set Transformer also introduces pooling by multihead attention. With seed
vectors $s_m$:

```math
z_m
=
\operatorname{Attn}(s_m,\widetilde{H}_i,\widetilde{H}_i).
```

Inducing-point variants reduce all-pairs cost by routing interaction through a
smaller learned set of inducing states.

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
