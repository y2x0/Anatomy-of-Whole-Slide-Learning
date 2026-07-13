# Set Transformer Versus Attention MIL

ABMIL and Set Transformer both use attention, but they put attention in
different mathematical roles.

## ABMIL

ABMIL computes:

```math
a_{ij}
=
\mathrm{softmax}_j s_\theta(h_{ij}),
\qquad
z_i
=
\sum_j a_{ij}v_\theta(h_{ij}).
```

There is no pairwise context before pooling unless the patch encoder already
encoded it.

Placement:

```text
C:
    instance scoring and value map

R:
    one weighted first moment
```

## Set Transformer

Set Transformer computes:

```math
\widetilde H_i
=
\mathrm{SAB}(H_i),
```

then:

```math
Z_i
=
\mathrm{PMA}(S,\widetilde H_i).
```

Placement:

```text
C:
    permutation-equivariant all-pairs set attention

R:
    learned seed-query readout
```

## Key Difference

ABMIL attention weights are functions of individual states:

```math
a_{ij}
=
f(h_{ij};H_i)
```

where dependence on
```math
H_i
```
comes through softmax normalization.

Set Transformer states are functions of all instances:

```math
\widetilde h_{ij}
=
f(h_{ij},H_i).
```

The readout sees interaction-aware states.

## Surviving Statistic

ABMIL preserves:

```math
z_i
=
\mathbb{E}_{j\sim a_i}[v(h_{ij})].
```

Set Transformer can preserve:

```math
Z_i
=
\left(
\mathbb{E}_{j\sim a_{i1}}[v(\widetilde h_{ij})],
\ldots,
\mathbb{E}_{j\sim a_{iK}}[v(\widetilde h_{ij})]
\right).
```

Thus Set Transformer can preserve multiple learned summaries of interaction
states.

## Dense Summary

ABMIL asks:

```text
which raw instances should be weighted?
```

Set Transformer asks:

```text
which interaction-aware set statistics should learned seeds extract?
```
