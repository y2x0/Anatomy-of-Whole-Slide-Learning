# Kernel MMD And Sketches

Kernel distribution pooling embeds the empirical measure into a feature space.

Let:

```math
\varphi(u)
\in
\mathbb{R}^{M}
```

be a kernel feature map or learned sketch. The pooled representation is:

```math
z_i
=
\frac{1}{n_i}\sum_j\varphi(u_{ij}).
```

## MMD View

For a kernel $k$, the squared MMD between two slide distributions is:

```math
\mathrm{MMD}^2(\mu_i,\mu_k)
=
\mathbb{E}_{u,u'\sim\mu_i}k(u,u')
+
\mathbb{E}_{v,v'\sim\mu_k}k(v,v')
-
2\mathbb{E}_{u\sim\mu_i,v\sim\mu_k}k(u,v).
```

If:

```math
k(u,v)
\approx
\varphi(u)^\top\varphi(v),
```

then:

```math
\mathrm{MMD}^2(\mu_i,\mu_k)
\approx
\|z_i-z_k\|_2^2.
```

## Finite Sketch

A sketch compresses the distribution:

```math
\mu_i
\mapsto
z_i
=
\mathbb{E}_{\mu_i}[\varphi(u)].
```

The choice of $\varphi$ decides which differences survive. Random features,
learned features, and prototype assignments are all sketching maps.

## Computational Cost

Direct pairwise MMD costs:

```math
O(n_i^2+n_k^2+n_i n_k).
```

Sketch pooling costs:

```math
O(n_iM)
```

to compute $z_i$, then:

```math
O(M)
```

to compare two slides.

## C/R/G/S Placement

```text
G:
    kernel geometry

C:
    feature map or random sketch

R:
    empirical kernel mean or sketch average

S:
    none for fixed kernels; task labels for learned sketches
```

## Dense Summary

Kernel/sketch pooling is moment pooling with a distribution-sensitive feature
map:

```math
\boxed{
z_i
=
\mathbb{E}_{\mu_i}[\varphi(u)]
}
```

Its value comes from choosing $\varphi$ so that task-relevant distribution
differences become linear differences in $z_i$.

