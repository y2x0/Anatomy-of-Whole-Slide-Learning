# Contrastive Supervision Objects

Contrastive learning observes relation labels.

Let an encoder produce:

```math
u_a
=
f_\theta(x_a)
\in
\mathbb{R}^{d}.
```

A contrastive dataset defines positive pairs:

```math
\mathcal{P}
\subset
\mathcal{A}\times\mathcal{A},
```

and candidate negatives:

```math
\mathcal{N}(a)
\subset
\mathcal{A}.
```

The supervision signal is:

```math
S^{\mathrm{obs}}
=
(\mathcal{P},\mathcal{N}).
```

## View Positives

Self-supervised positives are two augmentations of the same object:

```math
x_a^{(1)}
=
t_1(x_a),
\qquad
x_a^{(2)}
=
t_2(x_a).
```

The positive relation is:

```math
x_a^{(1)}
\sim
x_a^{(2)}.
```

This assumes augmentations preserve the latent semantic variable:

```math
U(t_1(x))
=
U(t_2(x)).
```

## Slide-Level Positives

Supervised contrastive WSI methods may define:

```math
(i,k)\in\mathcal{P}
\quad\Longleftrightarrow\quad
Y_i=Y_k.
```

This assumes same-label slides should be close in representation space.

For heterogeneous classes, this can be too strong.

## Patch-Level Positives

Patch positives may come from:

```text
two augmentations of same patch
patches in same cluster
patches in same region
patches with same pseudo-label
patches from same slide
```

Each defines a different latent equivalence relation.

## Relation Matrix

A contrastive supervision matrix is:

```math
M_{ab}
=
\mathbf{1}\{(a,b)\in\mathcal{P}\}.
```

Contrastive learning tries to make:

```math
u_a^\top u_b
```

large when
```math
M_{ab}=1
```
and small for sampled negatives.

## C/R/G/S Placement

```text
G:
    may define positive pairs through region, coordinate, or graph proximity

C:
    encoder learns invariances implied by positives

R:
    contrastive object may be patch, region, slide, prototype, or text embedding

S:
    relation labels M_ab
```

## Dense Summary

Contrastive supervision answers:

```text
What should be considered the same?
```

The loss then builds a representation geometry around that answer.
