# Mean As First Moment

Let a slide produce instance states:

```math
\widetilde H_i
=
\{u_{ij}\}_{j=1}^{n_i},
\qquad
u_{ij}\in\mathbb{R}^{d}.
```

The empirical measure is:

```math
\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\delta_{u_{ij}}.
```

Mean pooling computes:

```math
z_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}u_{ij}
=
\int u\,d\mu_i(u).
```

Thus mean pooling is the first moment of the empirical instance distribution.

## Permutation Invariance

For any permutation
```math
\pi
```
:

```math
\frac{1}{n_i}
\sum_{j=1}^{n_i}u_{ij}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}u_{i\pi(j)}.
```

Mean pooling ignores order by construction. It can be valid only when the
downstream target depends on composition rather than arrangement, or when
arrangement has already been encoded into
```math
u_{ij}
```
.

## What A Linear Head Sees

Suppose the task head is linear:

```math
\widehat y_i
=
w^\top z_i+b.
```

Then:

```math
\widehat y_i
=
w^\top
\left(
\frac{1}{n_i}\sum_j u_{ij}
\right)
b
=
\frac{1}{n_i}\sum_j w^\top u_{ij}+b.
```

The model averages instance evidence:

```math
s_{ij}=w^\top u_{ij}.
```

So the slide score is:

```math
\widehat y_i
=
\mathbb{E}_{j\sim\mathrm{Unif}(1,n_i)}[s_{ij}]+b.
```

Mean pooling with a linear head is exactly prevalence-weighted average evidence.

## Nonlinear Head Does Not Restore Lost Information

A nonlinear head:

```math
\widehat y_i
=
\mathcal{H}(z_i)
```

can learn nonlinear functions of the mean, but it cannot recover information not
present in the mean.

If:

```math
\frac{1}{n_i}\sum_j u_{ij}
=
\frac{1}{n_k}\sum_j u_{kj},
```

then:

```math
\mathcal{H}(z_i)
=
\mathcal{H}(z_k)
```

for every head
```math
\mathcal{H}
```
.

The bottleneck is the statistic, not the head.

## Moment Collision

Two different distributions can share the same mean:

```math
\int u\,d\mu_i(u)
=
\int u\,d\mu_k(u),
\qquad
\mu_i\ne\mu_k.
```

Example in one dimension:

```math
\mu_i
=
\frac{1}{2}\delta_{-1}
+
\frac{1}{2}\delta_{1},
```

and:

```math
\mu_k
=
\delta_{0}.
```

Both have mean zero:

```math
\int u\,d\mu_i(u)
=
\int u\,d\mu_k(u)
=
0.
```

Mean pooling cannot distinguish a bimodal slide from a homogeneous middle slide
if their first moments match.

## Mean Of Learned Features

A learned mean feature map uses:

```math
z_i
=
\frac{1}{n_i}
\sum_j\phi_\theta(u_{ij}).
```

This is still a mean, but of transformed features:

```math
z_i
=
\mathbb{E}_{\mu_i}[\phi_\theta(u)].
```

The transform can make useful morphology counts visible. For example, if one
coordinate of
```math
\phi_\theta(u)
```
approximates a tumor indicator, the mean estimates
tumor prevalence:

```math
z_{im}
\approx
\frac{\#\text{tumor-like patches}}{n_i}.
```

The surviving statistic remains a first moment in learned feature space.

This should not be conflated with the canonical Deep Sets representation, which
uses a sum of transformed instances. A mean can reproduce a sum only when the
cardinality `n_i` is fixed, supplied separately, or otherwise recoverable by the
downstream model. With variable slide sizes, normalized mean pooling and Deep
Sets sum pooling have different information content.

## Mean Versus Sum

Mean:

```math
z_i^{\text{mean}}
=
\frac{1}{n_i}\sum_j u_{ij}
```

estimates prevalence or average state.

Sum:

```math
z_i^{\text{sum}}
=
\sum_j u_{ij}
```

preserves total burden.

They differ by:

```math
z_i^{\text{sum}}
=
n_i z_i^{\text{mean}}.
```

If
```math
n_i
```
carries biological or sampling information, mean pooling removes it.
If
```math
n_i
```
is mostly nuisance tissue area or tiling density, mean pooling can be
the safer statistic.

## Dense Summary

```math
\boxed{
\mathcal{R}_{\text{mean}}(\widetilde H_i)
=
\mathbb{E}_{\mu_i}[u]
}
```

Mean pooling preserves the first moment and destroys everything not identifiable
from that first moment.
