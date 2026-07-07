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

If negatives are sampled from the marginal:

```math
k^-\sim P(k),
```

then the probability of class collision is:

```math
P(Y_{k^-}=Y_i)
=
\sum_c
P(Y_i=c)^2.
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
