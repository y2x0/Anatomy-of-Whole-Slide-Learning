# False Positives And Sampling Bias

Contrastive losses depend on correct positive and negative construction.

## False Negative

A false negative occurs when:

```math
k\in\mathcal{N}(i)
```

is treated as negative but shares the same latent class or morphology:

```math
U_k
=
U_i.
```

The loss pushes them apart:

```math
u_i^\top u_k
\downarrow.
```

This is common in pathology when two different slides contain the same tissue
pattern but are sampled as batch negatives.

## False Positive

A false positive occurs when:

```math
(i,k)\in\mathcal{P}
```

but the selected positive relation is not semantically valid.

Examples:

```text
same slide but different tissue regions
same class but different subtype
same cluster but different diagnosis
same report label but different morphology
```

The loss pulls unrelated objects together:

```math
u_i^\top u_k
\uparrow.
```

## Class Collision

For a fixed anchor with class
```math
Y_i=c
```
, a negative sampled from the marginal
collides with the anchor class with probability:

```math
P(Y_{k^-}=c\mid Y_i=c)
=
P(Y=c).
```

Before conditioning on the anchor, the average collision probability is:

```math
P(Y_{k^-}=Y_i)
=
\sum_c
P(Y=c)^2.
```

This class-level formula is still too weak for pathology. The more relevant
collision is latent morphology:

```math
P(U_{k^-}\sim U_i\mid Y_i=c),
```

where
```math
U
```
may include tissue state, grade, organ, stain, treatment effect, or
subtype. Two objects can collide morphologically even when their observed labels
are different:

```math
Y_{k^-}
\ne
Y_i,
\qquad
U_{k^-}\sim U_i.
```

Likewise, two objects can share the observed label while representing different
latent mechanisms:

```math
Y_{k^-}
=
Y_i,
\qquad
U_{k^-}\not\sim U_i.
```

In imbalanced datasets, majority classes create many false negatives.

## Batch Bias

The InfoNCE denominator estimates a population contrast using the batch:

```math
\sum_{k\in\mathcal{B}}
\exp(u_i^\top u_k/\tau).
```

If batch sampling is site-, class-, patient-, or organ-biased, the contrastive
task changes.

## Augmentation Violation

Self-supervised positives assume:

```math
U(t_1(x))
=
U(t_2(x)).
```

If augmentations remove diagnostic signal, alter stain in a label-changing way,
or crop out lesion evidence, the positive pair becomes invalid.

## Dense Summary

Contrastive supervision is only as correct as:

```math
\mathcal{P}
\quad
\text{and}
\quad
\mathcal{N}.
```

The loss is mathematically elegant, but the supervision comes from pair
construction. In WSI, pair construction is a biological assumption.
