# Concordance

Concordance evaluates whether predicted risk orders patients correctly.

Let $r_i$ be a scalar risk score. A pair $(i,j)$ is comparable if:

```math
X_i<X_j,
\qquad
\delta_i=1.
```

Then subject $i$ had an observed event before subject $j$ was known to fail
or be censored.

## Harrell C-Index

The empirical concordance is:

```math
\widehat{C}_{\text{Harrell}}
=
\frac{
\sum_{i,j}
\mathbf{1}[X_i<X_j,\delta_i=1]
\mathbf{1}[r_i>r_j]
}{
\sum_{i,j}
\mathbf{1}[X_i<X_j,\delta_i=1]
}.
```

Ties are often counted as one half:

```math
\mathbf{1}[r_i>r_j]
+
\frac{1}{2}\mathbf{1}[r_i=r_j].
```

## What Concordance Sees

Concordance sees:

```text
relative ordering
```

It does not see:

```text
calibration
survival curve shape
time-specific risk
absolute event probability
whether errors occur at clinically important horizons
```

A model can have good C-index and bad probability estimates.

## Uno IPCW C-Index

Harrell's estimator can be biased under heavy censoring. Uno-style concordance
uses inverse probability of censoring weights.

Let:

```math
\widehat{G}(t)=\Pr(C>t)
```

estimated from training data. A schematic IPCW estimator is:

```math
\widehat{C}_{\text{Uno}}(\tau)
=
\frac{
\sum_{i,j}
\mathbf{1}[X_i<X_j,X_i<\tau,\delta_i=1]
\widehat{G}(X_i)^{-2}
\mathbf{1}[r_i>r_j]
}{
\sum_{i,j}
\mathbf{1}[X_i<X_j,X_i<\tau,\delta_i=1]
\widehat{G}(X_i)^{-2}
}.
```

The truncation $\tau$ avoids regions where censoring weights become unstable.

## Survival-Curve Models Need A Risk Functional

If a model outputs $\widehat{S}_i(t)$, concordance still requires:

```math
r_i=\rho[\widehat{S}_i].
```

Examples:

```math
r_i=1-\widehat{S}_i(t^\star),
```

```math
r_i=-\int_0^{\tau}\widehat{S}_i(t)\,dt,
```

```math
r_i=-\mathrm{median}_{\widehat{S}_i}(T).
```

Different $\rho$ can produce different C-indices.

## WSI Consequence

WSI survival papers often report C-index because cohorts are small and
calibration is hard. That is acceptable as a discrimination metric, but it does
not validate:

```text
attention heatmaps
time-specific probabilities
clinical risk thresholds
external calibration
event-type-specific risk
```

## Dense Summary

```math
\widehat{C}
=
\frac{\#\text{correctly ordered comparable pairs}}
{\#\text{comparable pairs}}.
```

Concordance answers:

```text
are earlier-event patients assigned higher scalar risk?
```

It does not answer whether the survival model is probabilistically correct.
