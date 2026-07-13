# Hierarchical MIL C/R/G/S Design Matrix

## 1. Unified forward map

Let H_i^{(0)} be fine-scale features and P_i^{(\ell)} parent maps. A generic
hierarchical MIL model is

```math
H_i^{(\ell+1)}
=
\mathcal R_{\ell,\theta}
\left(
\mathcal C_{\ell,\theta}
\left(
H_i^{(\ell)},G_i^{(\ell)}
\right),
P_i^{(\ell)}
\right),
\qquad
\ell=0,\ldots,L-1,
```

followed by

```math
z_i
=
\mathcal R_{\mathrm{slide},\theta}
\left(
H_i^{(L)},G_i^{(L)}
\right),
\qquad
\widehat y_i
=
\mathcal H_{\omega}(z_i).
```

Here C is within-level context, R is fine-to-coarse or slide readout, and G
contains the geometry or support relation. The factorization separates the
operator from the object it acts on.

## 2. Paper placements

```text
Method       C                          R                         G
--------------------------------------------------------------------------------
DTFD-MIL     gated attention in        MaxS, MaxMinS, MAS,        random
             Tier 1 and Tier 2         or AFS, then attention    pseudo-bag
HIPT         MHSA plus FFN at          [CLS] extraction at       nested spatial
             each token scale          each scale                windows
HACT         GNN at cell and           assignment sum, then     cell graph,
             tissue levels             tissue graph readout     tissue graph,
                                                                  B matrix
```

The table deliberately distinguishes DTFD's random partition from HIPT's spatial
nesting and HACT's explicit graph hierarchy.

## 3. Surviving-statistic placement

```text
Method       Boundary statistic       Slide statistic             Main loss
--------------------------------------------------------------------------------
DTFD-MIL     selected patch or        weighted pseudo-bag         slide BCE plus
             Tier-1 attention mean    first moment                 Tier-1 BCE
HIPT         [CLS] token              transformer slide token     DINO plus
                                                                  downstream task
HACT         cell-state sum per       tissue-node sum or          slide CE
             tissue parent            layerwise sum-concat         or task loss
```

For DTFD, MaxS and MAS are sparse order-statistic-like choices while AFS is a
weighted first moment. For HIPT, [CLS] is a learned nonlinear summary rather
than a literal mean. For HACT, the cell-to-tissue sum preserves a
cellularity-sensitive quantity before the tissue graph.

## 4. Representation equivalences and non-equivalences

A hierarchy can reduce to a flat set model in special cases. If every parent
contains one child, P is a permutation and coarsening loses no child. If the
coarse context is absent and the final readout is an invariant set operator,
the hierarchy may be algebraically redundant.

It is not generally equivalent to a flat set model when:

```text
- parent identity changes the support of context;
- the coarsener is nonlinear or selective;
- fine coordinates are used before coarsening;
- coarse tokens are the only inputs to the task head;
- the number of children affects sum-scale features.
```

A distribution summary is also not automatically equivalent to a hierarchy. A
measure over features can retain global frequency information while discarding
which features share a parent. Conversely, a hierarchy can retain local
co-occurrence but discard fine-level distributions at each boundary.

## 5. Design rules for new models

A mathematically legible proposal should expose:

```math
\left(
\mathcal C_\ell,
\mathcal R_\ell,
P^{(\ell)},
\mathcal L_\ell
\right)_{\ell=0}^{L-1},
\qquad
\mathcal R_{\mathrm{slide}},
\qquad
\mathcal H_{\omega}.
```

Then answer these questions:

```text
1. Is P fixed, random, coordinate-derived, graph-derived, or learned?
2. Does C see only child features, or also parent geometry?
3. Is R a sum, mean, attention, top-k, prototype, or [CLS] summary?
4. What perturbations lie in the nullspace of each boundary?
5. Which labels supervise intermediate levels?
6. Does the final task require density, rarity, arrangement, or long-range
   co-occurrence?
```

The strongest new methods are not those with the most levels. They are those
whose hierarchy matches the statistic required by the task.

## 6. Final C/R/G/S interpretation

```text
C — how units interact before they are compressed:
    attention, transformer, graph message passing, or state-space context.

R — how children become parents and parents become a slide:
    sum, mean, attention, selection, prototype occupancy, or [CLS].

G — what makes a hierarchy a hierarchy:
    parent maps, coordinates, grid nesting, graphs, or learned routing.

S — where the signal comes from:
    slide labels, inherited pseudo-bag labels, self-supervised DINO views,
    cell or tissue supervision, or survival likelihood.

The C/R/G/S decomposition is not a taxonomy label pasted after training. It is a
factorization of the forward map that makes the model's inductive bias and
failure mode inspectable.
```
