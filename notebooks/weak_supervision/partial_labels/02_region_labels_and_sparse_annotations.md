# Region Labels And Sparse Annotations

Often supervision is not a patch label but a region label.

Let regions be:

```math
B_{ir}
\subset
\{1,\ldots,n_i\},
\qquad
r=1,\ldots,R_i.
```

A region label is:

```math
Y_{ir}^{\mathrm{reg}}
=
\Gamma_r(\{Z_{ij}:j\in B_{ir}\}).
```

The map $\Gamma_r$ may be OR, burden, majority, or pathologist-defined.

## Region OR Label

For lesion-present annotation:

```math
Y_{ir}^{\mathrm{reg}}
=
\max_{j\in B_{ir}}Z_{ij}.
```

The region likelihood is:

```math
P_\theta(Y_{ir}^{\mathrm{reg}}=1\mid H_i)
=
1-
\prod_{j\in B_{ir}}(1-p_{ij}).
```

This is MIL inside the region.

## Region Burden Label

If a region label means prevalence:

```math
B_{ir}
=
\frac{1}{|B_{ir}|}
\sum_{j\in B_{ir}}Z_{ij}.
```

A continuous burden annotation $b_{ir}$ can be modeled with:

```math
\widehat b_{ir}
=
\frac{1}{|B_{ir}|}
\sum_{j\in B_{ir}}p_{ij},
```

and:

```math
\mathcal{L}_{\mathrm{burden}}
=
\sum_{i,r}
(\widehat b_{ir}-b_{ir})^2.
```

## Sparse Point Annotation

A point annotation at coordinate $a_{im}$ may label only a local neighborhood:

```math
\mathcal{N}(a_{im})
=
\{j:\|c_{ij}-a_{im}\|\le r\}.
```

The point can be converted into a constraint:

```math
\max_{j\in\mathcal{N}(a_{im})}Z_{ij}
=
1.
```

or a soft target:

```math
w_{ijm}
\propto
\exp
\left(
-
\frac{\|c_{ij}-a_{im}\|_2^2}{2\sigma^2}
\right).
```

Then:

```math
\mathcal{L}_{\mathrm{point}}
=
-
\sum_{j}
w_{ijm}\log p_{ij}.
```

## Combining Region And Slide Labels

A joint objective can be:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{slide}}
+
\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}
+
\lambda_{\mathrm{inst}}\mathcal{L}_{\mathrm{inst}}.
```

This makes supervision multilevel:

```math
S_i
=
(Y_i,\{Y_{ir}^{\mathrm{reg}}\}_{r\in\mathcal{O}_i},\text{points}).
```

## C/R/G/S Placement

```text
G:
    regions and annotation coordinates

C:
    context can be local to annotated regions

R:
    region-level bag maps and slide-level bag maps coexist

S:
    supervision partially observes aggregate truth at intermediate scale
```

## Dense Summary

Region labels are not patch labels. They are smaller bag labels:

```math
Y_{\mathrm{region}}
=
\Gamma_{\mathrm{region}}(Z_{\mathrm{children}}).
```

The math should specify the region bag map before treating the annotation as
instance supervision.
