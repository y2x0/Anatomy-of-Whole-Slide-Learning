# Latent Instance Labels And Bag Map

A WSI bag has instance embeddings:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i}.
```

The ideal instance labels are:

```math
Z_i
=
(Z_{i1},\ldots,Z_{in_i}),
\qquad
Z_{ij}\in\{0,1\}.
```

In bag-label MIL,
```math
Z_i
```
is unobserved. The observed label is:

```math
Y_i
=
\Gamma(Z_i),
```

for some bag map:

```math
\Gamma:\{0,1\}^{n_i}\to\{0,1\}.
```

Different MIL assumptions are different choices of
```math
\Gamma
```
.

## Standard OR Map

The classic MIL map is:

```math
\Gamma_{\mathrm{OR}}(Z_i)
=
\max_j Z_{ij}.
```

Thus:

```math
Y_i=1
\quad\Longleftrightarrow\quad
\exists j:Z_{ij}=1.
```

The feasible latent-label set for a positive bag is:

```math
\mathcal{Z}_{+}(i)
=
\left\{
z\in\{0,1\}^{n_i}:
\sum_{j=1}^{n_i}z_j\ge 1
\right\}.
```

The feasible latent-label set for a negative bag is:

```math
\mathcal{Z}_{-}(i)
=
\{(0,\ldots,0)\}.
```

So negative bags are strongly informative; positive bags are weakly
informative.

## Threshold Map

A threshold MIL map is:

```math
\Gamma_{\tau}(Z_i)
=
\mathbf{1}
\left\{
\frac{1}{n_i}
\sum_{j=1}^{n_i}Z_{ij}
\ge
\tau
\right\}.
```

This is appropriate when the slide label depends on disease burden, not merely
one positive patch.

## Count Map

If the target is count-like:

```math
B_i
=
\sum_{j=1}^{n_i}Z_{ij}.
```

The observed label may be a discretization:

```math
Y_i
=
\mathbf{1}\{B_i\ge b_0\}.
```

Additive MIL and burden-style models are closer to this map than to the OR map.

## Distribution Map

Let the latent patch-label distribution be:

```math
\widehat\nu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\delta_{(h_{ij},Z_{ij})}.
```

A distributional bag label is:

```math
Y_i
=
F(\widehat\nu_i).
```

This includes labels driven by prevalence, mixture composition, heterogeneity,
or morphology distribution.

## C/R/G/S Placement

```text
G:
    can constrain which instances interact but does not reveal Z

C:
    creates features from which latent Z may be inferred

R:
    implements or approximates Gamma

S:
    observes only Y = Gamma(Z)
```

## Dense Summary

Bag-label MIL is underdetermined because:

```math
Y_i
=
\Gamma(Z_i)
```

is many-to-one. The loss observes the image of
```math
Z_i
```
 under
```math
\Gamma
```
, not
```math
Z_i
```
itself.
