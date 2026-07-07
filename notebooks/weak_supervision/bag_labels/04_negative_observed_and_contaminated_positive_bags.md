# Negative-Observed And Contaminated-Positive Bags

Standard OR-MIL does not create ordinary positive-unlabeled learning.

It creates a sharper asymmetry:

```text
negative-bag patches:
    observed negative under the OR assumption

positive-bag patches:
    contaminated mixture of positive and negative instances
```

## Negative Bags

Under:

```math
Y_i
=
\max_j Z_{ij},
```

negative bags identify negative instances:

```math
Y_i=0
\quad\Longrightarrow\quad
Z_{ij}=0
\quad
\forall j.
```

This is stronger than ordinary PU learning, where unlabeled examples contain
both positives and negatives and clean negatives may not be observed.

## Positive Bags

Positive bags contain at least one positive instance:

```math
Y_i=1
\quad\Longrightarrow\quad
\sum_{j=1}^{n_i}Z_{ij}\ge 1.
```

But each particular patch is still unlabeled:

```math
P(Z_{ij}=1\mid Y_i=1,H_i)
\in
[0,1].
```

Positive-bag patches are therefore contaminated positives, not clean positives.

## Two Negative Distributions

Let:

```math
P_1(h)
=
P(h\mid Z=1,Y=1),
```

```math
P_0^{+}(h)
=
P(h\mid Z=0,Y=1),
```

```math
P_0^{-}(h)
=
P(h\mid Z=0,Y=0).
```

The clean simplification:

```math
P_0^{+}
=
P_0^{-}
```

is an exchangeability assumption. It can fail in WSI because negative-looking
tissue inside positive slides may differ from tissue in negative slides by
organ, sampling, stain, cohort, inflammation, necrosis, or clinical workup.

## Positive-Bag Mixture

For a positive bag:

```math
P(h\mid Y=1)
=
\pi P_1(h)
+
(1-\pi)P_0^{+}(h),
```

where:

```math
\pi
=
P(Z=1\mid Y=1).
```

The naive simplification:

```math
P(h\mid Y=1)
=
\pi P_1(h)+(1-\pi)P_0(h)
```

is valid only if $P_0^{+}=P_0^{-}=P_0$.

## Naive Instance Training Bias

If every patch from a positive bag is labeled positive:

```math
\widetilde Z_{ij}
=
Y_i,
```

then the positive training distribution is:

```math
P(h\mid \widetilde Z=1)
=
\pi P_1(h)+(1-\pi)P_0^{+}(h).
```

The model may learn to separate $P_0^{+}$ from $P_0^{-}$ rather than $P_1$ from
negative tissue.

## Relation To PU Learning

True PU learning has:

```text
positive set:
    clean positives

unlabeled set:
    mixture of positives and negatives
```

OR-MIL bag labels have:

```text
negative bags:
    clean negatives under the model assumption

positive bags:
    contaminated positives
```

So the correct analogy is contaminated-label or negative-observed MIL, not
vanilla PU.

## C/R/G/S Placement

```text
G:
    can help separate true positives from positive-bag background

C:
    learns instance embeddings under asymmetric constraints

R:
    maps contaminated latent instances to a bag event

S:
    negative bags give clean negative constraints; positive bags give weak mixture constraints
```

## Dense Summary

The safe statement is:

```math
Y=0
\Rightarrow
Z_j=0
\quad
\forall j,
```

but:

```math
Y=1
\not\Rightarrow
Z_j=1.
```

Calling this ordinary PU hides the most important MIL asymmetry.
