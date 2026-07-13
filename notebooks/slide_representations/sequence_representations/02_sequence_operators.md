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
z=\mathrm{Pool}(u_1,\ldots,u_n).
```

The state
```math
s_j
```
summarizes previous tokens.

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
\mathrm{Attn}(H)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V.
```

A sequence transformer usually adds positional encoding:

```math
H'=H+P.
```

Without
```math
P
```
, self-attention is permutation equivariant, not sequence-aware.

With full attention, every token can interact with every other token. The
interaction graph is complete; the sequence enters through
```math
P
```
or through
relative position terms, not through a path constraint.

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

This is path-structured context. Information from token
```math
k
```
can affect token
```math
j
```
only through the intervening scan states.

The readout can be:

```math
z=\mathrm{Pool}(u_1,\ldots,u_n)
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

It does not make the representation permutation invariant. If the order changes,
the forward and backward paths change too.

## Dense Summary

```text
RNN:
    sequential hidden state

Transformer:
    complete-graph token interaction plus positional encoding

State-space:
    path-structured recurrence with long-context memory

Bidirectional scan:
    reduces direction artifact
```

All sequence operators inherit the chosen order's assumptions.
