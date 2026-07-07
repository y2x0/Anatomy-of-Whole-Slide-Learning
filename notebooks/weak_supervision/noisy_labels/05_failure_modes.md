# Failure Modes

Noisy-label methods fail when the noise model is wrong or the model mistakes
noise for signal.

## Unknown Transition Matrix

Risk correction requires:

```math
T_{ab}
=
P(\widetilde Y=b\mid Y=a).
```

If $T$ is estimated poorly:

```math
\widehat T
\ne
T,
```

then corrected losses can be more biased than ordinary losses.

## Instance-Dependent Noise

If:

```math
P(\widetilde Y\mid Y,H,G)
\ne
P(\widetilde Y\mid Y),
```

then a fixed transition matrix cannot represent the corruption channel.

Report-derived pathology labels often behave this way because ambiguity depends
on morphology, site, and clinical context.

## Memorization Of Noise

High-capacity WSI models can fit noisy labels:

```math
F_\theta(H_i,G_i)
=
\widetilde Y_i
```

even when:

```math
\widetilde Y_i
\ne
Y_i.
```

Weak labels plus high-dimensional features create many shortcut paths.

## Robust Filtering Removes Rare Truth

Small-loss selection can discard:

```text
rare subtype
subtle positive slide
out-of-distribution but valid morphology
hard survival case
```

because these produce high loss early.

## Shortcut Amplification

If noise is correlated with site:

```math
\widetilde Y
\leftarrow
\text{site}
\leftarrow
H,G,
```

the model may learn site-specific morphology or geometry rather than pathology.

## Dense Summary

Noisy labels require asking:

```text
Is the label wrong at random, wrong by class, wrong by morphology, or wrong by cohort?
```

Those are different statistical problems. Treating them as one generic "label
noise" problem hides the actual failure mode.
