# SAB And ISAB Context Before Pooling

Set Transformer separates context from readout.

The context stage produces:

```math
\widetilde H_i
=
\mathcal{C}_{\text{set}}(H_i).
```

The pooling stage reads:

```math
Z_i
=
\mathrm{PMA}(S,\widetilde H_i).
```

## Set Attention Block

A set attention block applies self-attention over all instances:

```math
\widetilde h_{ij}
=
\sum_{\ell=1}^{n_i}
\alpha_{ij\ell}Vh_{i\ell},
```

with:

```math
\alpha_{ij\ell}
=
\mathrm{softmax}_{\ell}
\left(
\frac{(Qh_{ij})^\top(Kh_{i\ell})}{\sqrt{d_q}}
\right).
```

This is permutation equivariant:

```math
\mathrm{SAB}(PH)
=
P\mathrm{SAB}(H).
```

After this context step, PMA can pool interaction-aware states.

## Induced Set Attention Block

Full self-attention costs:

```math
O(n_i^2).
```

ISAB introduces
```math
M
```
inducing points:

```math
I
=
\{i_1,\ldots,i_M\}.
```

First, inducing points attend to the set:

```math
A_i
=
\mathrm{MHA}(I,H_i,H_i).
```

Then set elements attend to the induced states:

```math
\widetilde H_i
=
\mathrm{MHA}(H_i,A_i,A_i).
```

The cost becomes:

```math
O(n_iM)
```

up to constants and heads.

## What Survives?

SAB/ISAB belongs to
```math
\mathcal{C}
```
, not
```math
\mathcal{R}
```
. The final readout still
decides what survives:

```math
Z_i
=
\mathrm{PMA}(S,\widetilde H_i).
```

The difference is that each
```math
\widetilde h_{ij}
```
may already contain pairwise or
global set context.

## C/R/G/S Placement

```text
G:
    complete learned relation graph, optionally approximated by inducing points

C:
    SAB or ISAB

R:
    PMA or invariant pooling over contextualized states

S:
    task loss shaping attention and seed queries
```

## Dense Summary

Set Transformer pooling is not just "attention pooling." It is:

```math
H_i
\xrightarrow{\mathrm{SAB/ISAB}}
\widetilde H_i
\xrightarrow{\mathrm{PMA}}
Z_i.
```

Context makes states interaction-aware. PMA decides which learned statistics of
those states survive.
