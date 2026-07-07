# Coordinates As Marks

A coordinate-aware slide is a marked point cloud:

```math
X_i
=
\{(h_{ij},c_{ij})\}_{j=1}^{n_i}.
```

The patch embedding is the mark:

```math
h_{ij}\in\mathbb{R}^{d},
```

and the coordinate is the point location:

```math
c_{ij}\in\Omega_i\subset\mathbb{R}^{2}.
```

## Coordinate-Augmented Features

The simplest coordinate injection concatenates position:

```math
u_{ij}
=
\phi_\theta
\left(
[h_{ij},\eta(c_{ij})]
\right),
```

where $\eta$ is a coordinate encoding.

Then any pooling operator acts on $u_{ij}$:

```math
z_i
=
\mathcal{R}(\{u_{ij}\}_{j=1}^{n_i}).
```

The model remains permutation invariant to input order, but not invariant to
coordinate changes.

## Coordinate-Dependent Attention

Attention can use coordinates in the score:

```math
s_{ij}
=
g_\theta(h_{ij},\eta(c_{ij})).
```

Then:

```math
a_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{\ell=1}^{n_i}\exp(s_{i\ell})},
\qquad
z_i
=
\sum_{j=1}^{n_i}a_{ij}v_\theta(h_{ij}).
```

This lets the model learn location-dependent weighting.

## Translation Symmetry

Absolute coordinates break translation invariance:

```math
f(\{(h_j,c_j+a)\}_j)
\ne
f(\{(h_j,c_j)\}_j)
```

in general.

If translation should be irrelevant, coordinate injection should use centered
coordinates or relative geometry:

```math
\bar c_i
=
\frac{1}{n_i}\sum_{j=1}^{n_i}c_{ij},
\qquad
\tilde c_{ij}
=
c_{ij}-\bar c_i.
```

But centering changes the meaning of location. It preserves relative layout
while removing the slide's absolute coordinate frame.

## Spatial Moment View

Coordinate-aware pooling can preserve spatial moments:

```math
m_i^{(h)}
=
\frac{1}{n_i}\sum_{j=1}^{n_i}h_{ij},
```

```math
m_i^{(c)}
=
\frac{1}{n_i}\sum_{j=1}^{n_i}c_{ij},
```

```math
M_i^{(hc)}
=
\frac{1}{n_i}\sum_{j=1}^{n_i}h_{ij}c_{ij}^{\top}.
```

The cross moment $M_i^{(hc)}$ captures where embedding dimensions tend to occur.
It is still a coarse statistic; it does not fully encode layout.

## C/R/G/S Placement

```text
G:
    coordinate marks C_i

C:
    feature map can depend on h_j and c_j

R:
    pooling can preserve coordinate-conditioned statistics

S:
    slide labels decide whether coordinate dependence is useful or shortcut-like
```

## Dense Summary

Coordinate geometry changes the slide from:

```math
\{h_j\}_{j=1}^{n}
```

to:

```math
\{(h_j,c_j)\}_{j=1}^{n}.
```

The model can now distinguish slides with the same morphology multiset but
different spatial locations.
