# Max As Extreme Statistic

Let each instance have a scalar evidence score:

```math
s_{ij}
=
g_\theta(u_{ij}).
```

Max pooling computes:

```math
r_i
=
\max_{1\le j\le n_i}s_{ij}.
```

This is permutation invariant:

```math
\max_j s_{ij}
=
\max_j s_{i\pi(j)}
```

for every permutation $\pi$.

## Extreme-Value Functional

Using the empirical measure:

```math
\mu_i
=
\frac{1}{n_i}
\sum_j\delta_{u_{ij}},
```

max pooling is:

```math
r_i
=
\sup_{u\in\mathrm{supp}(\mu_i)}g_\theta(u).
```

It depends on the support extreme, not the average mass.

Mean pooling asks:

```text
how much evidence is present on average?
```

Max pooling asks:

```text
does any instance contain enough evidence?
```

## Positive-Instance MIL

The classic positive-instance assumption is:

```math
y_i=1
\quad
\Longleftrightarrow
\quad
\exists j:\ y_{ij}=1.
```

If $s_{ij}$ estimates instance positivity, then:

```math
\max_j s_{ij}
```

is the natural bag-level score.

This matches tasks where one small region can determine the slide label:

```text
micrometastasis
small malignant focus
rare diagnostic organism
focal high-grade component
```

## Not A Distribution Moment

Max is not a first moment:

```math
\max_j s_{ij}
\ne
\frac{1}{n_i}\sum_j s_{ij}.
```

It ignores how many high-scoring patches exist once the largest one is known.

For:

```math
s_i=(10,0,0,0)
```

and:

```math
s_k=(10,10,10,10),
```

max pooling gives:

```math
\max s_i
=
\max s_k
=
10.
```

The slides are equivalent under max even though one has isolated evidence and
the other has diffuse evidence.

## Vector Max

If instance states are vectors:

```math
u_{ij}\in\mathbb{R}^{d},
```

componentwise max pooling is:

```math
z_{im}
=
\max_j u_{ijm}.
```

This creates a synthetic vector whose coordinates may come from different
patches:

```math
z_i
=
(u_{ij_1,1},u_{ij_2,2},\ldots,u_{ij_d,d}).
```

Componentwise max can preserve whether each feature appears somewhere, but it
does not preserve whether the features co-occur in the same patch.

## Gradient Through Hard Max

If:

```math
j^\star
=
\mathrm{argmax}_{j}s_{ij},
```

then:

```math
\frac{\partial r_i}{\partial s_{ij}}
=
\mathbf{1}\{j=j^\star\}
```

when the maximizer is unique.

Only the winning instance receives gradient through the max. This can sharpen
localization, but it can also make training unstable when early winners are
wrong.

## Dense Summary

```math
\boxed{
\mathcal{R}_{\max}(\widetilde H_i)
=
\sup_{u\in\mathrm{supp}(\mu_i)}g_\theta(u)
}
```

Max pooling preserves strongest evidence and discards prevalence, multiplicity,
and most non-winning instances.
