# Gradient-Times-Input And Taylor Credit

## Definition

For scalar target `F(x)`:

```math
\phi_j^{\mathrm{GxI}}
=
x_j
\frac{\partial F(x)}
{\partial x_j}.
```

Relative to zero baseline, first-order Taylor expansion gives:

```math
F(x)
\approx
F(0)
+
\sum_j
x_j
\frac{\partial F(x)}
{\partial x_j}.
```

Gradient-times-input is therefore local linear credit under a zero reference.

## Exactness For Homogeneous Functions

If `F` is differentiable and positively homogeneous of degree `k`:

```math
F(ax)
=
a^kF(x),
```

Euler's theorem gives:

```math
\sum_j
x_j
\frac{\partial F(x)}
{\partial x_j}
=
kF(x).
```

For degree one and zero bias:

```math
\sum_j
\phi_j^{\mathrm{GxI}}
=
F(x).
```

Biases, normalization, softmax, and nonhomogeneous activations break this exact
conservation.

## Baseline Shift

For baseline `x_0`, a more explicit local term is:

```math
\phi_j^{\mathrm{local}}
=
\left(
x_j-x_{0,j}
\right)
\frac{\partial F(x)}
{\partial x_j}.
```

Changing feature centering changes the attribution even when the represented
physical image is unchanged.

## Sign

```math
\phi_j^{\mathrm{GxI}}>0
```

means the current feature value and local score derivative align. It does not
mean the feature is intrinsically positive evidence across its full range.

## Interaction Allocation

For:

```math
F(x_1,x_2)
=
x_1x_2,
```

gradient-times-input gives:

```math
\phi_1
=
x_1x_2,
\qquad
\phi_2
=
x_1x_2.
```

The sum double-counts the interaction:

```math
\phi_1+\phi_2
=
2F.
```

Local Taylor terms do not uniquely allocate interaction credit.

## Patch Embeddings

For patch vector `h_j`, an instance score can sum featurewise terms:

```math
E_j^{\mathrm{GxI}}
=
\sum_{d=1}^{D}
h_{jd}
\frac{\partial F(H)}
{\partial h_{jd}}.
```

This yields signed patch-level local relevance while retaining context through
the full derivative. It remains dependent on feature parameterization and
zero-embedding semantics.
