# Transport And Geometry-Aware Pooling

Histograms treat bins or prototypes as categories. Transport adds geometry
between categories.

Let prototype prevalence be:

```math
p_i
\in
\Delta^{M-1}.
```

Let prototype cost be:

```math
C_{mn}
=
d(c_m,c_n)^2.
```

The optimal transport distance between two slides is:

```math
W_C(p_i,p_k)
=
\min_{T\ge 0}
\sum_{m,n}T_{mn}C_{mn}
```

subject to:

```math
\sum_n T_{mn}=p_{im},
\qquad
\sum_m T_{mn}=p_{kn}.
```

## Transport As Pooling Geometry

The pooled object may still be:

```math
z_i=p_i.
```

But the comparison geometry is not Euclidean:

```math
d(i,k)
=
W_C(p_i,p_k).
```

So the readout and the metric together define what survives.

## Entropic Transport

Entropic regularization solves:

```math
W_{C,\epsilon}(p_i,p_k)
=
\min_{T\ge 0}
\sum_{m,n}T_{mn}C_{mn}
+
\epsilon
\sum_{m,n}T_{mn}(\log T_{mn}-1)
```

with the same marginal constraints.

The regularized problem is smoother and computationally easier, but the
regularization blurs transport plans.

## Geometry-Aware Readout

Instead of storing only
```math
p_i
```
, one can store geometry-aware moments:

```math
z_i
=
\sum_{m=1}^{M}p_{im}c_m.
```

or residuals:

```math
r_i
=
\sum_m p_{im}(c_m-\bar c_i)(c_m-\bar c_i)^\top.
```

These are moments of the prototype distribution.

## C/R/G/S Placement

```text
G:
    prototype metric or transport cost C

C:
    assignment to prototype mass p_i

R:
    histogram plus geometry-aware metric or transport statistic

S:
    prototype learning objective and downstream task loss
```

## Dense Summary

Transport pooling is useful when:

```text
not all prototype mistakes are equally bad
```

It says that moving mass between similar morphologies should cost less than
moving mass between distant morphologies.
