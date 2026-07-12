# Attention Reweighting And Patch Deletion

## Deleting One Patch

For softmax attention over logits `e_j`, delete patch `i` while holding the
remaining logits and values fixed. The new weights are:

```math
\alpha_j^{(-i)}
=
\frac{
\alpha_j
}{
1-\alpha_i
}
\qquad
j\ne i.
```

The new pooled representation is:

```math
z_{-i}
=
\sum_{j\ne i}
\frac{\alpha_j}{1-\alpha_i}
v_j.
```

Using:

```math
z
=
\alpha_iv_i
+
\sum_{j\ne i}
\alpha_jv_j,
```

we obtain:

```math
z_{-i}
=
\frac{
z-\alpha_iv_i
}{
1-\alpha_i
}.
```

## Exact Representation Change

```math
z-z_{-i}
=
\frac{
\alpha_i
}{
1-\alpha_i
}
\left(
v_i-z
\right).
```

Removal effect depends on attention and deviation of the patch value from the
current pooled value.

If:

```math
v_i=z,
```

then:

```math
z-z_{-i}=0
```

even for high attention.

## Linear-Logit Removal

For linear class head:

```math
F_c(H)-F_c(H_{-i})
=
\frac{
\alpha_i
}{
1-\alpha_i
}
w_c^{\top}
\left(
v_i-z
\right).
```

Raw attention is insufficient because the sign and magnitude depend on
class-aligned deviation from the bag mean.

## Deletion Is Not Zeroing

Zeroing the value while retaining the attention denominator gives:

```math
z_i^{(0)}
=
z-\alpha_iv_i.
```

This differs from true instance deletion:

```math
z_i^{(0)}
\ne
z_{-i}
```

unless special conditions hold. WSI perturbation tests must state which
operation they use.

## Recomputed Context

The derivation above freezes all remaining scores and values. In self-attention,
graphs, or context encoders, deleting patch `i` changes:

```math
e_j,
\quad
v_j
\qquad
\text{for }j\ne i.
```

The exact effect then includes context recomputation and cannot be inferred
from the original attention vector.

## Cardinality Confounding

Patch deletion changes bag cardinality and normalization support. A score drop
can arise from missing tissue evidence, changed normalization, or out-of-support
bag size. These mechanisms need separate controls.
