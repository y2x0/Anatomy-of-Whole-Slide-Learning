# Interpretability Failure Modes

Attention maps are easy to show and easy to overclaim.

## Bright Patch Fallacy

The largest weight:

```math
j^\star
=
\arg\max_j a_j
```

is not necessarily the strongest evidence patch:

```math
j^\star
\ne
\arg\max_j a_jw^\top v_j.
```

High attention can select context, background, or negative evidence.

## Attention-Value Cancellation

Two patches can have high attention with opposite value evidence:

```math
a_1w^\top v_1
\approx
-
a_2w^\top v_2.
```

The heat map shows both as important, but their signed contributions cancel.

## Layer Ambiguity

In transformers:

```math
A^{(1)},A^{(2)},\ldots,A^{(L)}
```

are many attention maps. Choosing one layer or one head for visualization is a
post-hoc decision unless justified.

## Shortcut Maps

Attention can highlight:

```text
folds
pen marks
tissue edges
scanner artifacts
stain intensity
```

if those features reduce training loss. A visually localized map is not
automatically pathology-localized.

## Instability

If small augmentations change attention:

```math
a(H)
\not\approx
a(T(H)),
```

then the explanation is not stable. Accuracy can remain high while attention
maps shift.

## Dense Summary

A responsible attention explanation should report:

```text
which layer/head/readout
attention entropy and support
signed evidence or gradients
stability under perturbation
counterfactual deletion/insertion
pathologist-facing tissue validation
```

Without these, attention is a visualization of the computation, not a validated
explanation.

