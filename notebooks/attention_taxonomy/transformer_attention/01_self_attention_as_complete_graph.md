# Self-Attention As Complete Graph

Self-attention updates each token using all tokens as possible sources.

For token states:

```math
H
=
\{h_u\}_{u=1}^{n},
```

compute:

```math
Q
=
HW_Q,
\qquad
K
=
HW_K,
\qquad
V
=
HW_V.
```

Scores are:

```math
S
=
\frac{QK^\top}{\sqrt{d_k}}.
```

Weights are:

```math
A
=
\mathrm{softmax}_{\mathrm{row}}(S).
```

Output is:

```math
H'
=
AV.
```

## Complete Directed Graph

The row `u` update is:

```math
h_u'
=
\sum_{v=1}^{n}a_{uv}V_v,
\qquad
V_v
=
(HW_V)_v.
```

This is message passing on the complete directed graph over token indices:

```math
E
=
\mathcal{V}\times\mathcal{V},
\qquad
\mathcal{V}
=
\{1,\ldots,n\}.
```

Every token can send a message to every other token in one layer.

The graph statement concerns only the support of the attention sublayer. It
does not mean that the model learned a pathology graph, and it does not include
the residual connection, output projection, normalization, or feed-forward
subnetwork that may surround this sublayer.

## Permutation Equivariance

Without positional information, self-attention is permutation equivariant. For a
permutation matrix `P`:

```math
\mathrm{Attn}(PH)
=
P\mathrm{Attn}(H).
```

A readout that is also permutation invariant can ignore slide geometry entirely
unless coordinates or positional biases are added.

## WSI Cost

Full self-attention costs:

```math
O(n^2d).
```

For WSI patch counts:

```math
n
\sim
10^4
\text{ to }
10^5,
```

full attention is often infeasible without token reduction, windows, sampling,
hierarchy, linear attention, or state-space alternatives.

## Dense Summary

Transformer attention is:

```math
\boxed{
H'
=
\mathrm{softmax}
\left(
\frac{HW_QW_K^\top H^\top}{\sqrt{d_k}}
\right)
HW_V
}
```

The mathematical object is a learned dense stochastic adjacency matrix over
tokens.
