# Sequence Operators

Once a slide is represented as:

```math
(h_1,\ldots,h_n),
```

the model can use sequence operators.

## Recurrent Operator

A recurrent model computes:

```math
s_j=f_\theta(s_{j-1},h_j),
\qquad
u_j=g_\theta(s_j).
```

Readout:

```math
z=s_n
```

or:

```math
z=\operatorname{Pool}(u_1,\ldots,u_n).
```

The state $s_j$ summarizes previous tokens.

## Transformer Operator

Self-attention computes:

```math
Q=HW_Q,
\qquad
K=HW_K,
\qquad
V=HW_V.
```

Then:

```math
\operatorname{Attn}(H)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V.
```

A sequence transformer usually adds positional encoding:

```math
H'=H+P.
```

Without $P$, self-attention is permutation equivariant, not sequence-aware.

## State-Space Operator

A linear state-space model:

```math
s_{j+1}=As_j+Bh_j,
\qquad
u_j=Cs_j+Dh_j.
```

Selective state-space models make the parameters input-dependent:

```math
A_j=A(h_j),
\qquad
B_j=B(h_j),
\qquad
C_j=C(h_j).
```

Then:

```math
s_{j+1}=A_js_j+B_jh_j.
```

The readout can be:

```math
z=\operatorname{Pool}(u_1,\ldots,u_n)
```

or the final state.

## Bidirectional Scans

A one-directional scan is order-asymmetric:

```math
s_j^{\rightarrow}=f(s_{j-1}^{\rightarrow},h_j).
```

A bidirectional model also computes:

```math
s_j^{\leftarrow}=f(s_{j+1}^{\leftarrow},h_j).
```

and combines:

```math
u_j=[s_j^{\rightarrow};s_j^{\leftarrow}].
```

For WSI, bidirectionality reduces dependence on arbitrary scan direction.

## Dense Summary

```text
RNN:
    sequential hidden state

Transformer:
    all-pairs token interaction plus positional encoding

State-space:
    linear-time recurrence with long-context memory

Bidirectional scan:
    reduces direction artifact
```

All sequence operators inherit the chosen order's assumptions.
