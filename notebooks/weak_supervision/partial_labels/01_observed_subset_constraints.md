# Observed Subset Constraints

Let latent instance labels be:

```math
Z_i
=
(Z_{i1},\ldots,Z_{in_i}).
```

A partial-label dataset observes labels only on a subset:

```math
\mathcal{O}_i
\subset
\{1,\ldots,n_i\}.
```

The observed supervision is:

```math
S_i^{\mathrm{obs}}
=
\{(j,Z_{ij}):j\in\mathcal{O}_i\}.
```

The unobserved set is:

```math
\mathcal{U}_i
=
\{1,\ldots,n_i\}\setminus\mathcal{O}_i.
```

## Partial Likelihood

If labels are observed exactly on
```math
\mathcal{O}_i
```
, the partial supervised
likelihood is:

```math
P_\theta(S_i^{\mathrm{obs}}\mid H_i)
=
\prod_{j\in\mathcal{O}_i}
P_\theta(Z_{ij}=s_{ij}\mid h_{ij}).
```

The corresponding loss is:

```math
\mathcal{L}_{\mathrm{partial}}
=
-
\sum_i
\sum_{j\in\mathcal{O}_i}
\log
P_\theta(Z_{ij}=s_{ij}\mid h_{ij}).
```

This ignores unlabeled instances unless combined with bag, consistency, or
regularization terms.

## Masked Risk

The target measure must be stated before writing a risk. A per-observed-patch
risk is not the same as a per-slide risk.

Let:

```math
M_{ij}
=
\mathbf{1}\{j\in\mathcal{O}_i\}.
```

The empirical masked risk is:

```math
\widehat R_{\mathrm{mask}}(\theta)
=
\frac{
\sum_i\sum_j M_{ij}\ell(g_\theta(h_{ij}),Z_{ij})
}{
\sum_i\sum_j M_{ij}
}.
```

This estimates the risk over the observed annotation distribution. It estimates
the full per-instance risk:

```math
R_{\mathrm{inst}}(\theta)
=
\mathbb{E}_{i,j}
\left[
\ell(g_\theta(h_{ij}),Z_{ij})
\right]
```

only if missingness is ignorable and the sampling measure over
```math
(i,j)
```
matches
the desired target measure.

## Missingness Assumption

The clean assumption is:

```math
M_{ij}
\perp
Z_{ij}
\mid
h_{ij}.
```

If annotation probability depends on the true label beyond observed features:

```math
P(M=1\mid Z,h)
\ne
P(M=1\mid h),
```

then the masked risk is biased.

## Inverse-Probability Weighting

If annotation probability is:

```math
\pi_{ij}
=
P(M_{ij}=1\mid h_{ij},G_i),
```

then a per-instance inverse-probability risk is:

```math
\widehat R_{\mathrm{IPW}}(\theta)
=
\frac{1}{N}
\sum_i
\frac{1}{n_i}
\sum_j
\frac{M_{ij}}{\pi_{ij}}
\ell(g_\theta(h_{ij}),Z_{ij}).
```

This is unbiased under correct
```math
\pi_{ij}
```
and positivity:

```math
\pi_{ij}>0.
```

If the target is per-slide rather than per-patch, the normalization changes. If
the target is region burden, the target measure changes again. The IPW formula
is not meaningful until that measure is named.

## C/R/G/S Placement

```text
G:
    annotation locations and spatial sampling scheme

C:
    instance encoder receives direct gradients on observed subset

R:
    bag readout may still be trained jointly

S:
    partial observation mask and observed labels
```

## Dense Summary

Partial labels are not just "some labels." They are labels plus a sampling
mechanism:

```math
S^{\mathrm{obs}}
=
(M,M\odot Z).
```

If the mask
```math
M
```
is biased, the supervision channel is biased.
