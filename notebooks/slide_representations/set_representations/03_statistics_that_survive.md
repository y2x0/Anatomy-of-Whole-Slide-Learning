# Statistics That Survive

Set representations force the slide into a permutation-invariant statistic.

The core question:

```text
What information about the empirical patch distribution survives?
```

## First Moment

Mean pooling:

```math
z_i=\frac{1}{n_i}\sum_{j=1}^{n_i}h_{ij}.
```

Survives:

```math
\mathbb{E}_{\mu_i}[h].
```

Lost:

```text
multimodality
rare subpopulations
spatial arrangement
higher-order interactions
```

## Transformed First Moment

Deep Sets:

```math
z_i=\sum_j\phi(h_{ij}).
```

Survives:

```math
\mathbb{E}_{\mu_i}[\phi(h)].
```

This can encode nonlinear morphology counts, but only after instance-wise
transformation.

## Extreme Statistic

Max MIL:

```math
z_i=\max_jg(h_{ij}).
```

Survives:

```text
largest patch-level evidence
```

Good for:

```text
small lesion detection
rare positive morphology
```

Bad for:

```text
diffuse phenotypes
noisy high-scoring patches
multi-region evidence
```

## Weighted First Moment

Attention:

```math
z_i=\sum_ja_{ij}v(h_{ij}),
\qquad
\sum_ja_{ij}=1.
```

Survives:

```math
\mathbb{E}_{j\sim a_i}[v(h_{ij})].
```

The attention distribution $a_i$ changes which parts of the empirical measure
matter.

## Distribution Summary

Prototype prevalence:

```math
p_{im}
=
\frac{1}{n_i}\sum_jq_m(h_{ij}).
```

Survives:

```math
p_i=(p_{i1},\ldots,p_{iM}).
```

This approximates the morphology distribution more explicitly than a single
mean.

## Pairwise Interactions Without Geometry

Set attention can compute:

```math
\widetilde{h}_j
=
\sum_k
\operatorname*{softmax}_{k}(q_j^\top k_k)v_k.
```

This is a complete-graph context operator:

```math
\mathcal{N}_{K_n}(j)=\{1,\ldots,n\}.
```

Every patch can send a learned message to every other patch before invariant
pooling.

Then invariant pooling:

```math
z=\sum_j\widetilde{h}_j.
```

Survives:

```text
interaction-aware set summaries
```

Still missing:

```text
which interactions were spatially adjacent unless coordinates are included
```

## Dense Summary

```text
mean:
    first moment

Deep Sets:
    transformed first moment

max:
    extreme evidence

attention:
    weighted transformed first moment

prototype:
    distribution over learned morphologies

set attention:
    interaction-aware invariant statistic
```

The slide-as-set view is mathematically clean, but every valid readout must
discard order and explicit geometry unless those are encoded as features.
