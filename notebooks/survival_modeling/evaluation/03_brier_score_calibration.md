# Brier Score And Calibration

The Brier score evaluates predicted survival probabilities, not just ordering.

At horizon
```math
t
```
, the event-free target is:

```math
Y_i^{S}(t)=\mathbf{1}[T_i>t].
```

The model predicts:

```math
\widehat{S}_i(t).
```

Without censoring:

```math
\mathrm{BS}(t)
=
\frac{1}{n}\sum_i
(Y_i^{S}(t)-\widehat{S}_i(t))^2.
```

## IPCW Brier Score

With right censoring, use censoring survival:

```math
\widehat{G}(t)=\Pr(C>t).
```

The IPCW Brier score is:

```math
\mathrm{BS}^{c}(t)
=
\frac{1}{n}
\sum_i
\left[
\frac{\mathbf{1}[X_i\le t,\delta_i=1]}{\widehat{G}(X_i)}
(0-\widehat{S}_i(t))^2
+
\frac{\mathbf{1}[X_i>t]}{\widehat{G}(t)}
(1-\widehat{S}_i(t))^2
\right].
```

Events before
```math
t
```
 have true survival status
```math
0
```
. Subjects observed beyond
```math
t
```
 have true survival status
```math
1
```
.

## Integrated Brier Score

For time range
```math
[t_1,t_2]
```
:

```math
\mathrm{IBS}
=
\frac{1}{t_2-t_1}
\int_{t_1}^{t_2}
\mathrm{BS}^{c}(t)\,dt.
```

Numerically:

```math
\mathrm{IBS}
\approx
\frac{1}{t_2-t_1}
\sum_{k}
\mathrm{BS}^{c}(t_k)\Delta t_k.
```

IBS evaluates the survival curve over time.

## Calibration

Calibration at time
```math
t
```
asks:

```math
\Pr(T>t\mid \widehat{S}(t)=q)=q.
```

Equivalently for event risk:

```math
\Pr(T\le t\mid \widehat{F}(t)=q)=q.
```

Empirical calibration under censoring requires IPCW or Kaplan-Meier estimates
within predicted-risk groups.

## Decomposition Intuition

Brier score combines:

```text
calibration
resolution
uncertainty
```

A model with strong ranking but overconfident probabilities can have a poor
Brier score.

## WSI Consequence

WSI survival papers that report only C-index do not show whether predicted
survival probabilities are usable for:

```text
risk thresholds
treatment decision support
clinical trial stratification
patient counseling
```

If a model outputs only Cox risk score
```math
\eta_i
```
, Brier score requires
recovering:

```math
\widehat{S}_i(t)
=
\exp[-\exp(\eta_i)\widehat{\Lambda}_0(t)].
```

If the model outputs discrete hazards, survival is:

```math
\widehat{S}_i(\tau_k)
=
\prod_{\ell\le k}(1-\widehat{h}_{i\ell}).
```

## Dense Summary

```math
\mathrm{BS}^{c}(t)
=
\frac{1}{n}
\sum_i
\left[
\frac{\mathbf{1}[X_i\le t,\delta_i=1]}{\widehat{G}(X_i)}
\widehat{S}_i(t)^2
+
\frac{\mathbf{1}[X_i>t]}{\widehat{G}(t)}
(1-\widehat{S}_i(t))^2
\right].
```

Brier score is the survival metric that forces the model to put probabilities
where its ranking claims are.
