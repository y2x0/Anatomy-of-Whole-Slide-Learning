# Integrated Gradients Axioms And Completeness

Primary anchor:

- Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML
  2017.

## Definition

For input `x`, baseline `x'`, and scalar function `F`:

```math
\mathrm{IG}_j(x;x')
=
\left(
x_j-x_j'
\right)
\int_0^1
\frac{
\partial F
\left(
x'+\alpha(x-x')
\right)
}{
\partial x_j
}
\,\mathrm{d}\alpha.
```

It integrates gradients along the straight path from baseline to input.

## Completeness Proof

Define:

```math
\gamma(\alpha)
=
x'+\alpha(x-x').
```

Then:

```math
\frac{\mathrm{d}}
{\mathrm{d}\alpha}
F
\left(
\gamma(\alpha)
\right)
=
\sum_j
\frac{
\partial F
\left(
\gamma(\alpha)
\right)
}{
\partial x_j
}
\left(
x_j-x_j'
\right).
```

Integrating gives:

```math
\sum_j
\mathrm{IG}_j(x;x')
=
F(x)-F(x').
```

Completeness conserves score difference relative to the baseline.

## Sensitivity

If `x` and `x'` differ in only coordinate `j` and:

```math
F(x)\ne F(x'),
```

then completeness implies:

```math
\mathrm{IG}_j(x;x')
\ne
0.
```

This addresses endpoint saturation that can make raw gradient zero.

## Implementation Invariance

If two networks implement the same differentiable function everywhere:

```math
F(x)=G(x),
```

their gradients agree almost everywhere, hence:

```math
\mathrm{IG}^{F}(x;x')
=
\mathrm{IG}^{G}(x;x').
```

Modified-backpropagation methods need not satisfy this.

## Numerical Approximation

With `m` Riemann steps:

```math
\widehat{\mathrm{IG}}_j
=
\left(
x_j-x_j'
\right)
\frac{1}{m}
\sum_{r=1}^{m}
\frac{
\partial F
\left(
x'+\frac{r}{m}(x-x')
\right)
}{
\partial x_j
}.
```

Completeness error is:

```math
\varepsilon_{\mathrm{comp}}
=
\left|
\sum_j
\widehat{\mathrm{IG}}_j
-
\left[
F(x)-F(x')
\right]
\right|.
```

This should be checked rather than assumed for finite `m`.
