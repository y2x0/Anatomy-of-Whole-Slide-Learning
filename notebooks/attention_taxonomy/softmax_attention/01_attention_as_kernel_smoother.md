# Attention As Kernel Smoother

Softmax attention converts compatibility scores into a probability distribution.

For query `q` and source states:

```math
\{(k_j,v_j)\}_{j=1}^{n},
```

define:

```math
s_j
=
\frac{q^\top k_j}{\sqrt{d_k}}.
```

The weights are:

```math
a_j
=
\frac{\exp(s_j)}
{\sum_{\ell=1}^{n}\exp(s_\ell)}.
```

The output is:

```math
m(q)
=
\sum_{j=1}^{n}a_jv_j.
```

## Exponential Kernel View

The score induces a positive kernel:

```math
K(q,k_j)
=
\exp
\left(
\frac{q^\top k_j}{\sqrt{d_k}}
\right).
```

Then attention is the normalized kernel smoother:

```math
m(q)
=
\frac{\sum_j K(q,k_j)v_j}
{\sum_j K(q,k_j)}.
```

Thus dot-product attention has the form of a Nadaraya-Watson estimator in a
learned feature space, conditional on the learned projections being fixed. The
learned part is not the normalization; it is the geometry in which similarity
is computed.

The word ``kernel'' needs a qualification here. The function

```math
K(q,k)
=
\exp
\left(
\frac{q^\top k}{\sqrt{d_k}}
\right)
```

is positive as a weighting function, but a general query-key score need not
define a symmetric positive-semidefinite kernel on one common domain. The
taxonomy therefore uses ``kernel smoother'' for the normalized weighting form,
not as a claim that every attention score is a reproducing-kernel construction.

## Additive Attention

Additive attention uses:

```math
s_j
=
w^\top
\tanh(W_q q + W_k k_j + b).
```

The same normalized-kernel form holds:

```math
K_\theta(q,k_j)
=
\exp(s_j).
```

The difference is the score family. Dot-product attention is bilinear after
projection. Additive attention passes the query-key pair through a nonlinear
compatibility network. In both cases, the exponential score is a positive
weighting function and the output is a normalized first moment of the values.

## MIL Readout As Degenerate Query Attention

In attention MIL, there may be no explicit query token. Scores are instance
local:

```math
s_j
=
w^\top\tanh(W_a h_j).
```

This can be written as query attention with a learned constant query:

```math
s_j
=
\psi_\theta(q_0,h_j),
\qquad
q_0
\text{ fixed or learned}.
```

The slide embedding is:

```math
z
=
\sum_{j=1}^{n}a_jv_\theta(h_j).
```

So ABMIL is not context attention. It is a learned global query that reads a bag
into one weighted first moment.

## What The Kernel Smooths Away

The output is inside the convex hull of values:

```math
m(q)
\in
\mathrm{conv}
\{v_1,\ldots,v_n\}.
```

For one readout token, all higher-order structure is represented only through
how the score changes the weights and how the value map changes the points.

The output cannot directly preserve:

```text
unweighted prevalence
multiple separated modes
pairwise co-occurrence
spatial topology
```

unless those facts are already encoded in the values or multiple queries are
used.

## Dense Summary

Softmax attention is:

```math
\boxed{
m(q)
=
\mathbb{E}_{j\sim a(q,H)}[v_j]
}
```

with:

```math
a(q,H)
=
\mathrm{softmax}
\left(
\psi_\theta(q,k_1),\ldots,\psi_\theta(q,k_n)
\right).
```

The inductive bias is learned similarity followed by a dense normalized average.
