# Sum Of Patch Evidence

Additive pooling assigns each instance an evidence score:

```math
e_{ij}
=
e_\theta(u_{ij}).
```

The bag score is:

```math
r_i
=
\sum_{j=1}^{n_i}e_{ij}.
```

The prediction may be:

```math
\widehat y_i
=
\mathrm{sigmoid}(r_i)
```

for binary classification, or:

```math
\eta_i
=
r_i
```

for a Cox-style risk score.

## Sum Versus Mean

Mean evidence pooling is:

```math
\bar r_i
=
\frac{1}{n_i}\sum_j e_{ij}.
```

Sum evidence pooling is:

```math
r_i
=
\sum_j e_{ij}.
```

Mean asks about average evidence density. Sum asks about total burden.

## Additive Decomposition

The key interpretability property is exact decomposition:

```math
r_i
=
e_{i1}+\cdots+e_{in_i}.
```

Removing patch
```math
k
```
changes the score by:

```math
r_i-r_i^{(-k)}
=
e_{ik}.
```

This is stronger than attention heatmaps, where removing a patch changes the
normalization and can alter all weights.

## Signed Evidence

Evidence can be positive or negative:

```math
e_{ij}>0:
    \text{supports the class},
```

```math
e_{ij}<0:
    \text{opposes the class}.
```

The final score is the net balance:

```math
r_i
=
\sum_{j:e_{ij}>0}e_{ij}
-
\sum_{j:e_{ij}<0}|e_{ij}|.
```

## C/R/G/S Placement

```text
G:
    absent unless u_ij includes spatial or graph context

C:
    evidence function e_theta

R:
    additive sum

S:
    slide label or survival loss that calibrates evidence units
```

## Dense Summary

Additive pooling is:

```math
\boxed{
r_i
=
\sum_j e_\theta(u_{ij})
}
```

It is interpretable because each patch has an exact signed contribution to the
bag score.
