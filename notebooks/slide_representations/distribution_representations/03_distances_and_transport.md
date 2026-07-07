# Distances And Transport

Once a slide is a distribution $\mu_i$, comparing slides becomes a problem of
comparing probability measures.

## Euclidean Distance Between Summaries

If:

```math
z_i=T(\mu_i),
```

then a simple distance is:

```math
d(i,k)
=
\|z_i-z_k\|_2.
```

This is cheap, but it only compares the statistic $T$. If $T$ discards
multimodality, the distance cannot recover it.

## Histogram Divergence

For prototype prevalence vectors:

```math
p_i,p_k\in\Delta^{M-1},
```

one can use:

```math
D_{\text{KL}}(p_i\|p_k)
=
\sum_{m=1}^{M}
p_{im}
\log
\frac{p_{im}}{p_{km}}.
```

or a symmetric variant:

```math
D_{\text{JS}}(p_i,p_k)
=
\frac{1}{2}D_{\text{KL}}(p_i\|m)
+
\frac{1}{2}D_{\text{KL}}(p_k\|m),
```

where:

```math
m=\frac{1}{2}(p_i+p_k).
```

These distances treat prototypes as categories unless ground geometry is added.

## Optimal Transport

Let prototype centers be $c_1,\ldots,c_M$. Define a transport cost:

```math
C_{mn}
=
\|c_m-c_n\|^2.
```

The optimal transport distance between two prototype histograms is:

```math
W_C(p_i,p_k)
=
\min_{T\in\mathbb{R}_{+}^{M\times M}}
\sum_{m,n}T_{mn}C_{mn}
```

subject to:

```math
\sum_n T_{mn}=p_{im},
\qquad
\sum_m T_{mn}=p_{kn}.
```

This compares morphology distributions while respecting similarity between
prototypes.

## Maximum Mean Discrepancy

For kernel mean embeddings:

```math
\mathrm{MMD}^2(\mu_i,\mu_k)
=
\left\|
m_{\mu_i}-m_{\mu_k}
\right\|_{\mathcal{H}}^2.
```

Expanding:

```math
\mathrm{MMD}^2
=
\mathbb{E}_{h,h'\sim\mu_i}k(h,h')
+
\mathbb{E}_{g,g'\sim\mu_k}k(g,g')
-
2\mathbb{E}_{h\sim\mu_i,g\sim\mu_k}k(h,g).
```

Empirically, this uses all pairwise kernel comparisons between patches in the
two slides.

For finite slides, the unbiased empirical estimate has the form:

```math
\widehat{\mathrm{MMD}}^2
=
\frac{1}{n_i(n_i-1)}
\sum_{a\ne b}
k(h_{ia},h_{ib})
+
\frac{1}{n_k(n_k-1)}
\sum_{a\ne b}
k(h_{ka},h_{kb})
-
\frac{2}{n_i n_k}
\sum_{a,b}
k(h_{ia},h_{kb}).
```

The direct cost is:

```math
O(n_i^2+n_k^2+n_i n_k),
```

which is usually too large for naive WSI retrieval. This is why practical
distribution distances often operate on prototypes, sketches, random features,
or subsampled patches rather than every tile embedding.

## Distributional Retrieval

Distribution distances can drive retrieval:

```math
(k_1,\ldots,k_K)
=
\mathrm{argsort}_{k\ne i}
\ d(\mu_i,\mu_k)
\quad
\text{ascending},
```

```math
\mathcal{N}_K(i)
=
\{k_1,\ldots,k_K\}.
```

The retrieved slides are those whose morphology distributions are closest under
the chosen metric.

The metric defines what "similar slide" means:

```text
mean distance:
    similar average morphology

histogram divergence:
    similar prevalence of prototypes

optimal transport:
    similar up to movement between related prototypes

MMD:
    similar kernel-level distribution shape
```

## Dense Summary

```math
\boxed{
\text{slide comparison}
=
\text{measure comparison}
}
```

Distribution representations make similarity mathematically explicit. The cost
is that the chosen statistic and metric decide which differences can be seen.
