# Gradient, Attention, And Evidence

Attention weight and predictive evidence can be combined.

For a readout:

```math
z
=
\sum_j a_jv_j,
```

and scalar output:

```math
o
=
g(z),
```

the first-order change from perturbing value `v_j` is:

```math
\delta o
\approx
\left(
\nabla_z g(z)
\right)^\top
a_j\delta v_j.
```

This is a value-space perturbation. It treats the attention weights as fixed.
By Cauchy-Schwarz, its magnitude obeys:

```math
|\delta o|
\le
a_j
\|\nabla_z g(z)\|_2
\|\delta v_j\|_2.
```

Thus a local sensitivity score is:

```math
e_j
=
a_j
\left\|
\nabla_z g(z)
\right\|
\left\|
\delta v_j
\right\|
```

only after a perturbation scale `delta v_j` has been specified. Taking
`delta v_j=-v_j` gives a perturbation-to-zero heuristic, not a universal
evidence definition. For a signed linear head, the exact value-path
contribution is instead:

```math
e_j
=
a_jw^\top v_j.
```

## Attention Times Logit Evidence

For class `c`:

```math
o_c
=
w_c^\top
\sum_j a_jv_j,
```

the patch-level signed contribution is:

```math
e_{jc}
=
a_j w_c^\top v_j.
```

This separates:

```text
looked at:
    a_j

supports class c:
    w_c^T v_j

contributes to class c:
    a_j w_c^T v_j
```

## Gradient Through Scores

Attention scores also affect output:

```math
\frac{\partial o}{\partial s_k}
=
\sum_j
\frac{\partial o}{\partial a_j}
\frac{\partial a_j}{\partial s_k}.
```

For a linear head:

```math
\frac{\partial o}{\partial a_j}
=
w^\top v_j.
```

Softmax gives:

```math
\frac{\partial a_j}{\partial s_k}
=
a_j(\mathbf{1}\{j=k\}-a_k).
```

So:

```math
\frac{\partial o}{\partial s_k}
=
a_k
\left(
w^\top v_k
-
\sum_j a_jw^\top v_j
\right).
```

A patch score increases the logit only if its value evidence exceeds the
attention-weighted average evidence.

## Gradient Through Patch Embeddings

For a patch embedding `h_j`, attention has at least two derivative paths:

```text
value path:
    h_j -> v_j -> z -> o

score path:
    h_j -> s -> a -> z -> o
```

For a single-query readout:

```math
z
=
\sum_{\ell}a_{\ell}(h)v_{\ell}(h),
\qquad
o
=
g(z),
```

the total derivative can be written schematically as:

```math
\frac{\partial o}{\partial h_j}
=
\left(
\frac{\partial v_j}{\partial h_j}
\right)^\top
a_j\nabla_z g(z)
+
\sum_k
\left(
\frac{\partial s_k}{\partial h_j}
\right)^\top
\frac{\partial o}{\partial s_k}.
```

For softmax attention and a linear head:

```math
\frac{\partial o}{\partial s_k}
=
a_k
\left(
w^\top v_k
-
\sum_j a_jw^\top v_j
\right).
```

Thus high attention is not the same as high derivative. A patch can matter by
having high value evidence, by changing the score normalization, or by changing
both.

## Dense Summary

Better evidence diagnostics use:

```text
attention mass
value alignment
head sensitivity
counterfactual removal
stability under perturbation
```

Attention alone is a measure. Evidence is a relation between the measure,
values, and head.
