# Marginal Likelihood And Posterior Ambiguity

Bag labels require marginalizing latent instance labels.

Let:

```math
p_{ij}
=
P_\theta(Z_{ij}=1\mid h_{ij}).
```

Assume conditional independence:

```math
P_\theta(Z_i=z\mid H_i)
=
\prod_{j=1}^{n_i}
p_{ij}^{z_j}(1-p_{ij})^{1-z_j}.
```

## OR Likelihood

For the OR bag map:

```math
Y_i
=
\max_j Z_{ij}.
```

The positive probability is:

```math
P_\theta(Y_i=1\mid H_i)
=
\sum_{z\in\mathcal{Z}_{+}(i)}
P_\theta(Z_i=z\mid H_i).
```

Because the complement is the all-negative event:

```math
P_\theta(Y_i=0\mid H_i)
=
\prod_{j=1}^{n_i}(1-p_{ij}),
```

we get:

```math
P_\theta(Y_i=1\mid H_i)
=
1-\prod_{j=1}^{n_i}(1-p_{ij}).
```

## Posterior Instance Probability

For a positive bag, the posterior probability that instance $j$ is positive is:

```math
P_\theta(Z_{ij}=1\mid Y_i=1,H_i)
=
\frac{
p_{ij}
}{
1-\prod_{\ell=1}^{n_i}(1-p_{i\ell})
}.
```

This is not normalized across instances. Multiple instances can be positive.

For a negative bag:

```math
P_\theta(Z_{ij}=1\mid Y_i=0,H_i)
=
0.
```

Thus positive bags have posterior ambiguity; negative bags do not under the
standard OR model.

## Bag Gradient

Let:

```math
P_i
=
P_\theta(Y_i=1\mid H_i)
=
1-\prod_j(1-p_{ij}).
```

For a positive bag loss:

```math
\mathcal{L}_i^{+}
=
-\log P_i.
```

The derivative with respect to $p_{ij}$ is:

```math
\frac{\partial \mathcal{L}_i^{+}}{\partial p_{ij}}
=
-
\frac{
\prod_{\ell\ne j}(1-p_{i\ell})
}{
1-\prod_{\ell}(1-p_{i\ell})
}.
```

For a negative bag:

```math
\mathcal{L}_i^{-}
=
-\sum_j\log(1-p_{ij}),
```

so:

```math
\frac{\partial \mathcal{L}_i^{-}}{\partial p_{ij}}
=
\frac{1}{1-p_{ij}}.
```

Negative bags push every instance down. Positive bags distribute credit through
posterior ambiguity.

## Non-Identifiability

Suppose two instances have identical features:

```math
h_{i1}
=
h_{i2}.
```

Then any symmetric model must assign:

```math
p_{i1}
=
p_{i2}.
```

But even when features differ, bag likelihood may not identify which one is the
true witness if both can explain the slide label.

Formally, if:

```math
1-\prod_j(1-p_{ij})
=
1-\prod_j(1-p'_{ij})
```

for every observed bag, then the bag likelihood cannot distinguish $p$ from
$p'$.

## Dense Summary

Bag-label training estimates:

```math
P(Y\mid H),
```

not:

```math
P(Z_j\mid H).
```

Instance explanations require extra assumptions, auxiliary losses, geometry,
pseudo-labels, or validation beyond the bag likelihood.
