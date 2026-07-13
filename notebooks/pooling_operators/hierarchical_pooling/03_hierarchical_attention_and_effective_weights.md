# Hierarchical Attention And Effective Weights

Nested attention produces effective patch weights.

Suppose region
```math
r
```
pools patches with weights:

```math
a_{ij\mid r},
\qquad
j\in\mathrm{Ch}(r).
```

The region state is:

```math
g_{ir}
=
\sum_{j\in\mathrm{Ch}(r)}
a_{ij\mid r}v_{ij}.
```

The slide pools regions with weights:

```math
z_i
=
\sum_r b_{ir}g_{ir}.
```

Substitute:

```math
z_i
=
\sum_r
b_{ir}
\sum_{j\in\mathrm{Ch}(r)}
a_{ij\mid r}v_{ij}.
```

Thus the effective patch weight is:

```math
w_{ij}^{\mathrm{eff}}
=
b_{ir(j)}a_{ij\mid r(j)},
```

where
```math
r(j)
```
 is the region containing patch
```math
j
```
.

## Unequal Region Sizes

If local pooling is mean and global pooling is mean:

```math
g_{ir}
=
\frac{1}{|\mathrm{Ch}(r)|}
\sum_{j\in\mathrm{Ch}(r)}u_{ij},
```

```math
z_i
=
\frac{1}{M_i}\sum_r g_{ir},
```

then:

```math
w_{ij}^{\mathrm{eff}}
=
\frac{1}{M_i|\mathrm{Ch}(r(j))|}.
```

Patches in small regions receive larger effective weight than patches in large
regions.

## Effective Measure

The hierarchy defines a new weighted empirical measure:

```math
\nu_i^{\mathrm{hier}}
=
\sum_j w_{ij}^{\mathrm{eff}}\delta_{u_{ij}}.
```

The slide embedding is:

```math
z_i
=
\int v(u)\,d\nu_i^{\mathrm{hier}}(u).
```

## Dense Summary

Hierarchical attention should always be reduced to effective weights when
debugging:

```math
w_{ij}^{\mathrm{eff}}
=
\text{product of weights along the path to the slide}.
```

This exposes whether a method is really focusing on tissue or merely weighting
regions unevenly.
