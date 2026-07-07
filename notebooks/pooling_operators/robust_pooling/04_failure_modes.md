# Robust Pooling Failure Modes

Robust pooling protects against outliers, but pathology often contains rare
truth.

## 1. Rare Positive Removal

If true disease evidence appears in the upper tail:

```math
s_{ij^\star}
\gg
s_{ij},
```

then trimming or clipping can remove the most important patch.

## 2. Artifact And Disease Look Similar

Robust pooling assumes extremes are suspect. But an extreme score can be:

```text
artifact
true rare lesion
scanner shortcut
diagnostic morphology
```

The operator cannot know which without better context or supervision.

## 3. Hidden Failure Under Good Average Performance

A robust average may improve cohort-level stability while hurting rare cases.
This can look good in aggregate metrics but fail clinically important sparse
phenotypes.

## 4. Coordinatewise Synthetic States

Coordinatewise medians and trimmed means can produce:

```math
z_i
\notin
\{u_{ij}\}
```

and not even near a plausible patch embedding.

## Dense Summary

Robust pooling should be used when:

```text
outliers are more likely to be artifacts than signal
```

It should be treated with suspicion when:

```text
the label is defined by rare positive evidence
```

