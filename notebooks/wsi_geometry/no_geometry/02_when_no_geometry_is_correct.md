# When No Geometry Is Correct

Ignoring geometry is not automatically naive. It is correct when the task is
approximately a function of the empirical morphology distribution.

## Distribution-Only Label

Let:

```math
\widehat\mu_i
=
\frac{1}{n_i}\sum_{j=1}^{n_i}\delta_{h_{ij}}.
```

If the target satisfies:

```math
y_i
=
F(\widehat\mu_i)
```

for some functional $F$, then coordinates are nuisance variables.

Examples of distribution-like targets include:

```text
morphology prevalence:
    how much of a phenotype is present

rare positive detection:
    whether at least one diagnostic patch exists

global tissue composition:
    relative abundance of tissue patterns

retrieval by morphology:
    similarity of visual content independent of arrangement
```

For these tasks, forcing spatial context may add variance without adding
signal.

## Sufficient Statistic View

Coordinates are unnecessary if:

```math
P(Y_i\mid H_i,C_i)
=
P(Y_i\mid H_i).
```

Equivalently:

```math
Y_i
\perp
C_i
\mid
H_i.
```

This conditional independence statement is the real assumption behind
no-geometry WSI learning.

## Morphology Distribution Functional

A model such as:

```math
z_i
=
\mathcal{R}(\{h_{ij}\}_{j=1}^{n_i}),
\qquad
\widehat y_i
=
\mathcal{H}(z_i)
```

is enough if $z_i$ preserves the morphology statistic needed by the task.

Mean pooling assumes:

```math
F(\widehat\mu_i)
\approx
\rho
\left(
\int h\,d\widehat\mu_i(h)
\right).
```

Prototype pooling assumes:

```math
F(\widehat\mu_i)
\approx
\rho
\left(
T_{\mathrm{proto}}(\widehat\mu_i)
\right).
```

Attention pooling assumes:

```math
F(\widehat\mu_i)
\approx
\rho
\left(
\int v_\theta(h)\,d\nu_{\theta,i}(h)
\right),
```

where $\nu_{\theta,i}$ is a label-trained reweighting of the empirical
distribution.

## C/R/G/S Placement

```text
G:
    intentionally ignored

C:
    learns patch morphology states

R:
    estimates a morphology statistic

S:
    teaches which morphology statistic predicts the task
```

The defensible version of no-geometry is not:

```text
coordinates are useless
```

It is:

```text
for this target, coordinates add little conditional information beyond morphology
```

## Dense Summary

No-geometry is correct when:

```math
Y
\perp
C
\mid
H.
```

It is wrong when the label depends on spatial relations not encoded in the patch
features.
