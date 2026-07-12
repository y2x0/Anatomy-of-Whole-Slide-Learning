# Attention Weight Versus Score Contribution

Primary anchor:

- Javed et al. "Additive MIL: Intrinsically Interpretable Multiple Instance
  Learning." NeurIPS 2022 Workshop.
  https://arxiv.org/abs/2206.01794

## Linear Head

For class logit:

```math
F_c(H)
=
w_c^{\top}z+b_c,
```

and attention readout:

```math
z
=
\sum_j
\alpha_jv_j,
```

the score decomposes as:

```math
F_c(H)
=
b_c
+
\sum_j
\alpha_j
w_c^{\top}v_j.
```

Define direct class contribution:

```math
c_{j,c}^{\mathrm{direct}}
=
\alpha_j
w_c^{\top}v_j.
```

Attention and contribution coincide up to a positive constant only if:

```math
w_c^{\top}v_j
=
\kappa_c
>
0
\qquad
\forall j.
```

This condition is rarely satisfied.

## Sign Failure

Because:

```math
\alpha_j\ge0,
```

attention cannot encode sign. Yet:

```math
w_c^{\top}v_j
<
0
```

makes a highly attended patch inhibitory for class `c`.

## Class Failure

For two classes:

```math
c_{j,a}
=
\alpha_jw_a^{\top}v_j,
```

```math
c_{j,b}
=
\alpha_jw_b^{\top}v_j.
```

One class-agnostic attention map can accompany opposite class contributions.

## Nonlinear Head

For:

```math
F_c(H)
=
g_c(z),
```

there is generally no exact additive decomposition. A first-order local term
is:

```math
c_{j,c}^{\mathrm{local}}
=
\alpha_j
\nabla_z g_c(z)^{\top}v_j.
```

This is a tangent approximation, not an exact finite contribution.

## Full Patch Derivative

If scores and values depend on patch `h_i`:

```math
\frac{\partial F_c}
{\partial h_i}
=
\left(
\frac{\partial v_i}
{\partial h_i}
\right)^{\top}
\alpha_i
\nabla_z g_c
+
\sum_j
\left(
\frac{\partial\alpha_j}
{\partial h_i}
\right)
v_j^{\top}
\nabla_z g_c.
```

The second term is attention redistribution and is invisible in the raw weight.

## Exact Equality Boundary

Raw attention is exact score credit only when the readout, value map, head, and
reference semantics make:

```math
F_c(H)-F_c(H_0)
=
K_c
\sum_j\alpha_j
```

with patch-specific credits proportional to `alpha_j`. Since normalized weights
sum to one, this would usually make the score independent of their distribution
unless values also enter. Attention alone cannot generally carry score magnitude.
