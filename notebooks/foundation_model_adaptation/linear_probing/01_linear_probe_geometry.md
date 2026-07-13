# Linear Probe Geometry

Let
```math
z_i\in\mathbb{R}^{D}
```
be a frozen slide representation. A binary linear
probe is:

```math
s_i
=
w^\top z_i+b,
\qquad
\widehat p_i
=
\mathrm{sigmoid}(s_i).
```

The decision boundary is a hyperplane:

```math
\{z:w^\top z+b=0\}.
```

## Separability

A dataset is linearly separable in frozen space if there exists
```math
(w,b)
```
such
that:

```math
y_i(w^\top z_i+b)>0
```

for labels
```math
y_i\in\{-1,1\}
```
. The margin is:

```math
\gamma
=
\min_i
\frac{y_i(w^\top z_i+b)}
{\|w\|_2}.
```

A large margin means the task direction is already clean in the foundation
space. A small or zero margin means either the task is not linearly encoded or
the cohort is noisy.

## What The Probe Cannot Do

If two slides share the same representation:

```math
z_i
=
z_k
```

but different labels:

```math
y_i
\ne
y_k,
```

then every probe gives the same score:

```math
w^\top z_i+b
=
w^\top z_k+b.
```

If classes are separated only by a nonlinear rule:

```math
y
=
\mathbf{1}\{z_1z_2>0\},
```

then a linear probe is the wrong adaptation class even if the representation is
informative.

## Dense Summary

Linear probing asks:

```math
Y
\perp
X
\mid
w^\top F_{\phi_0}(X)
?
```

It is a strong diagnostic of frozen linear readability, not a full measure of
foundation-model capacity.
