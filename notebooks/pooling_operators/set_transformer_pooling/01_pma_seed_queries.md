# PMA Seed Queries

Pooling by multihead attention uses learned seed vectors:

```math
S
=
\{s_1,\ldots,s_K\},
\qquad
s_k\in\mathbb{R}^{d}.
```

Given permutation-equivariant set states:

```math
\widetilde H_i
=
\{\widetilde h_{ij}\}_{j=1}^{n_i},
```

PMA computes:

```math
Z_i
=
\mathrm{MHA}(S,\widetilde H_i,\widetilde H_i).
```

For one head and one seed:

```math
z_{ik}
=
\sum_{j=1}^{n_i}
a_{ikj}V\widetilde h_{ij},
```

where:

```math
a_{ikj}
=
\frac{
\exp
\left(
\frac{(Qs_k)^\top(K\widetilde h_{ij})}{\sqrt{d_q}}
\right)
}{
\sum_{\ell}
\exp
\left(
\frac{(Qs_k)^\top(K\widetilde h_{i\ell})}{\sqrt{d_q}}
\right)
}.
```

## Difference From ABMIL

ABMIL computes a score from each instance:

```math
a_{ij}
\propto
\exp(s_\theta(h_{ij})).
```

PMA computes weights from a learned query:

```math
a_{ikj}
\propto
\exp(q_k^\top k_{ij}).
```

The seed vector asks a question of the set. Multiple seeds ask multiple
questions.

## Multi-Seed Readout

With
```math
K
```
seeds:

```math
Z_i
=
[z_{i1},\ldots,z_{iK}]
\in
\mathbb{R}^{K\times d}.
```

A task head can flatten or pool the seed outputs:

```math
\widehat y_i
=
\mathcal{H}_\theta(Z_i).
```

This lets the model preserve several statistics rather than one weighted first
moment.

## Permutation Invariance

If
```math
\widetilde H_i
```
is permutation equivariant and PMA attends over the set, then
the output seed set is invariant to input order:

```math
\mathrm{PMA}(S,P\widetilde H_i)
=
\mathrm{PMA}(S,\widetilde H_i).
```

The seed order is learned and fixed, but the input patch order is irrelevant.

## Dense Summary

PMA pooling is:

```math
\boxed{
Z_i
=
\mathrm{Attn}(\text{learned queries},\text{set states},\text{set states})
}
```

It generalizes attention MIL from one learned scoring rule to multiple learned
queries over contextualized set states.
