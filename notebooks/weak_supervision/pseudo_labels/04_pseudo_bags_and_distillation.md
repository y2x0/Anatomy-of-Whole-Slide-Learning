# Pseudo-Bags And Distillation

Some WSI methods create smaller bags from a large slide, train or score those
pseudo-bags, then distill information back into a slide model.

DTFD-MIL-style methods fit this broad pattern.

## Pseudo-Bag Partition

Partition slide instances into $M_i$ pseudo-bags:

```math
\{1,\ldots,n_i\}
=
\bigcup_{m=1}^{M_i}
B_{im},
\qquad
B_{im}\cap B_{im'}=\varnothing.
```

Each pseudo-bag has an embedding:

```math
z_{im}
=
\mathcal{R}_{\mathrm{local}}
(\{h_{ij}:j\in B_{im}\}).
```

The slide readout is:

```math
z_i
=
\mathcal{R}_{\mathrm{global}}
(\{z_{im}\}_{m=1}^{M_i}).
```

## Pseudo-Bag Label

The slide label may be assigned to each pseudo-bag:

```math
\widehat Y_{im}
=
Y_i.
```

This is noisy because a positive slide can contain negative pseudo-bags.

Alternatively, a teacher can score pseudo-bags:

```math
\widehat q_{im}
=
P_\phi(Y\mid B_{im}).
```

Then training uses:

```math
\mathcal{L}_{\mathrm{pb}}
=
\sum_{i,m}
\mathrm{CE}
(P_\theta(Y\mid B_{im}),\widehat q_{im}).
```

## Feature Distillation

If a local model computes pseudo-bag features:

```math
u_{im}
=
F_\phi(B_{im}),
```

the slide model can learn:

```math
z_i
=
\mathcal{R}_\theta(\{u_{im}\}_{m=1}^{M_i}).
```

The supervision is no longer only $Y_i$; it includes the intermediate
representation produced by the distillation process.

## Bias-Variance Tradeoff

Pseudo-bags reduce bag size:

```math
|B_{im}|
\ll
n_i.
```

This improves optimization and localizes evidence, but increases label noise:

```math
P(\widehat Y_{im}\ne Y_{im}^{\mathrm{true}})
```

can be high when slide labels are copied to all pseudo-bags.

## C/R/G/S Placement

```text
G:
    pseudo-bags may be random, clustered, spatial, or attention-defined

C:
    local pseudo-bag features are learned before global context

R:
    local and global readouts are coupled

S:
    slide labels are converted into pseudo-bag labels or teacher scores
```

## Dense Summary

Pseudo-bag methods trade one hard weak-supervision problem:

```math
Y_i
\text{ supervises }
n_i
\text{ patches}
```

for many smaller noisy problems:

```math
\widehat Y_{im}
\text{ supervises }
B_{im}.
```

The method succeeds when pseudo-bag construction increases signal concentration
more than it increases label noise.
