# Softmax Attention Failure Modes

Softmax attention is smooth and trainable, but the normalization creates
specific failure modes.

## Attention Collapse

If one score dominates:

```math
s_m-s_j
\gg
0
\quad
\text{for all }j\ne m,
```

then:

```math
a_m
\approx
1,
\qquad
a_j
\approx
0.
```

The slide statistic becomes:

```math
z
\approx
v_m.
```

The model behaves like max pooling over the learned score, even when disease
evidence is distributed.

## Attention Dilution

If a rare positive patch has score only slightly higher than many background
patches:

```math
s_+
=
s_0+\epsilon,
```

with `n` background patches at score `s_0`, then:

```math
a_+
=
\frac{\exp(\epsilon)}
{n+\exp(\epsilon)}.
```

For large bags:

```math
a_+
\approx
\frac{\exp(\epsilon)}{n}.
```

Rare evidence can vanish under the denominator.

More precisely, to retain a target mass `c` in `(0,1)`, the positive score
must satisfy:

```math
a_+
\ge
c
\quad\Longleftrightarrow\quad
\epsilon
\ge
\log
\left(
\frac{cn}{1-c}
\right).
```

The required score advantage therefore grows logarithmically with the number
of background patches. Increasing the bag size is not a neutral preprocessing
change for a fixed attention score scale.

## Shortcut Concentration

If artifact features correlate with labels, attention may learn:

```math
s_j
=
\psi_\theta(\text{artifact}_j)
```

instead of pathology. The weights may look visually plausible if artifacts are
near tissue, but the selected statistic is shortcut-driven.

## Non-Identifiability Of Weights

A logit can be written:

```math
o
=
w^\top
\sum_j a_jv_j
=
\sum_j a_jw^\top v_j.
```

The same logit can arise from different pairs:

```math
(a_j,v_j)
\quad
\text{and}
\quad
(\tilde a_j,\tilde v_j).
```

Thus attention weights alone do not identify evidence.

Even within one parameterization, weight and contribution are different
quantities. For a linear head:

```math
o
=
w^\top z
=
\sum_j a_j\,w^\top v_j,
```

a patch can receive large attention while contributing little to the output if
its projected value is nearly orthogonal to `w`. Conversely, a moderately
weighted patch can dominate the logit through its value direction.

## Bag-Size Dependence

Because all weights sum to one:

```math
\sum_j a_j
=
1,
```

duplicating low-score background can still affect calibration and gradients
through the denominator. Attention is not automatically invariant to tiling
density.

## Dense Summary

Softmax attention fails when:

```text
the correct statistic is sparse but score separation is weak
the correct statistic is distributed but attention collapses
bag size changes the normalization
high attention is mistaken for causal evidence
shortcuts receive high compatibility scores
```

The operator is smooth, but the survival statistic remains a normalized weighted
average.
