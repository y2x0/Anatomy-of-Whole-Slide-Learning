# Calibration And Domain Shift

A linear probe can rank slides well while producing poorly calibrated
probabilities.

## Logit Scale

For a logistic probe:

```math
\widehat p(z)
=
\mathrm{sigmoid}(w^\top z+b).
```

If logits are scaled:

```math
s'(z)
=
as(z),
\qquad
a>0,
```

the ranking is unchanged, but the probabilities change:

```math
\mathrm{sigmoid}(as)
\ne
\mathrm{sigmoid}(s).
```

Thus AUROC or C-index can look good while calibration is bad.

## Temperature Calibration

A calibrated probability can use:

```math
\widehat p_T(z)
=
\mathrm{softmax}(o(z)/T).
```

The temperature
```math
T
```
changes confidence without changing multiclass argmax when
```math
T>0
```
.

## Domain Shift

Let the train distribution be
```math
P_{\mathrm{tr}}(Z,Y)
```
and the deployment
distribution be
```math
P_{\mathrm{te}}(Z,Y)
```
. A probe trained under:

```math
P_{\mathrm{tr}}(Y\mid Z)
```

may be deployed under:

```math
P_{\mathrm{te}}(Y\mid Z).
```

If:

```math
P_{\mathrm{tr}}(Y\mid Z)
\ne
P_{\mathrm{te}}(Y\mid Z),
```

then no amount of frozen representation quality fixes calibration by itself.

## Dense Summary

Linear probing evaluates readability of a frozen geometry, but clinical use also
requires calibrated probabilities under the target cohort.
