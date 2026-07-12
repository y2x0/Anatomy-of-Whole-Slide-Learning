# CLAM Class-Specific Heatmap Map

Primary anchor:

- Lu et al. "Data-Efficient and Weakly Supervised Computational Pathology on
  Whole-Slide Images." Nature Biomedical Engineering 2021.
  https://arxiv.org/abs/2004.09666

## Class-Specific Attention

For patch feature `h_k`, CLAM computes gated hidden representation:

```math
q_k
=
\tanh
\left(
V_a h_k
\right)
\odot
\sigma
\left(
U_a h_k
\right).
```

For class branch `m`, unnormalized score is:

```math
e_{k,m}
=
W_{a,m}q_k.
```

Normalized class-specific attention is:

```math
a_{k,m}
=
\frac{
\exp(e_{k,m})
}{
\sum_{j=1}^{N}
\exp(e_{j,m})
}.
```

The class-specific slide representation is:

```math
z_m
=
\sum_{k=1}^{N}
a_{k,m}h_k.
```

## Heatmap Construction

For predicted class `m_hat`, the displayed patch heatmap is based on:

```math
\left\{
a_{k,\widehat m}
\right\}_{k=1}^{N}.
```

Spatial rendering maps each patch score back to its WSI coordinate and
interpolates or averages overlapping tiles.

## Relative Class-Conditional Ranking

The map answers:

```math
\text{which patches received the largest normalized readout weight in branch }
\widehat m.
```

It does not directly answer:

```math
\text{which patches have the largest signed contribution to class logit }
\widehat m.
```

The latter also depends on patch values and the class head.

## Branch Dependence

For one patch:

```math
a_{k,m}
\ne
a_{k,m'}
```

in general. A WSI has multiple class-conditional attention maps, not one
intrinsic importance map.

## Attention Extremes As Generated Supervision

For ground-truth class `Y`, CLAM sorts attention and selects top and bottom
instances. Abstractly:

```math
I_Y^{+}
=
\mathrm{TopK}
\left(
\left\{
a_{k,Y}
\right\},K
\right),
```

```math
I_Y^{-}
=
\mathrm{BottomK}
\left(
\left\{
a_{k,Y}
\right\},K
\right).
```

The auxiliary clustering loss encourages separation of these selected
features. This can sharpen heatmaps, but selection remains model-generated
rather than patch-annotated.

## Mutual-Exclusivity Assumption

Out-of-class branches can receive negative constraints under mutually
exclusive slide labels. Multi-label pathology violates this assumption when
multiple morphologies legitimately coexist.

## Proper Interpretive Claim

CLAM heatmaps visualize relative, class-specific attention used to construct
the slide representation. Their empirical overlap with tumor regions is a
localization result, not an algebraic identity between attention and tumor
probability.
