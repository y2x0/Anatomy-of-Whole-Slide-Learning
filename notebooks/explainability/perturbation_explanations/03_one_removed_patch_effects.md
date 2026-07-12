# One-Removed Patch Effects

## Definition

For patch `i`:

```math
\Delta_i^{\mathrm{LOO}}
=
F(X)
-
F(X_{-i}).
```

Positive effect means the patch supports the current scalar target under
deletion. Negative effect means removing it increases the target.

## Additive Model

If:

```math
F(X)
=
b
+
\sum_j e_j,
```

then:

```math
\Delta_i^{\mathrm{LOO}}
=
e_i.
```

One-removed effects are exact additive credits.

## Attention Model

For:

```math
z
=
\sum_j\alpha_jv_j,
\qquad
F(X)
=
w^{\top}z+b,
```

with remaining logits and values fixed after deletion:

```math
\Delta_i^{\mathrm{LOO}}
=
\frac{
\alpha_i
}{
1-\alpha_i
}
w^{\top}
\left(
v_i-z
\right).
```

The effect depends on how the patch differs from the existing bag summary.

## Interaction Example

Let:

```math
F
\left(
\left\{
x_1,x_2
\right\}
\right)
=
x_1x_2.
```

Then:

```math
\Delta_1^{\mathrm{LOO}}
=
x_1x_2,
\qquad
\Delta_2^{\mathrm{LOO}}
=
x_1x_2.
```

Their sum double-counts the interaction.

## Context Dependence

For the same patch in two bags:

```math
\Delta_i^{\mathrm{LOO}}(X)
\ne
\Delta_i^{\mathrm{LOO}}(X')
```

because the readout, competing tissue, and contextual representations differ.

## Computational Cost

For bag of `n` patches, exact leave-one-out requires:

```math
n+1
```

forward passes. With self-attention cost `Theta(n^2d)`, naive total cost is:

```math
\Theta
\left(
n^3d
\right).
```

This is prohibitive for tens of thousands of WSI patches.

## Explanation Boundary

One-removed effects answer finite necessity under single-patch deletion. They
do not allocate interactions uniquely or identify which larger tissue region is
sufficient.
