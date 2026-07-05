# Cox Family Failure Modes

The Cox family is clean because it removes the baseline hazard from training.
That same move creates several failure modes.

## 1. Non-Proportional Hazards

Cox assumes:

```math
\frac{\lambda(t\mid z_i)}{\lambda(t\mid z_j)}
=
\exp(\eta_i-\eta_j).
```

The hazard ratio is constant over time.

This fails when a feature changes meaning across time. In cancer prognosis,
early recurrence and late recurrence may be driven by different biology.

A WSI may contain:

```text
aggressive invasive morphology relevant to early death
immune microenvironment patterns relevant to later survival
treatment-response morphology that changes with time window
```

A scalar Cox score forces all of these into one ordering.

## 2. Ranking Without Calibration

The partial likelihood learns relative order. It does not directly learn:

```math
S(t\mid z_i)
```

unless the baseline hazard is estimated afterward.

Two models can have similar concordance but very different survival
calibration. This matters if the goal is clinical risk over a time horizon,
not just ranking patients.

## 3. Risk-Set Bias Under Cohort Shift

Each event is compared against its risk set:

```math
R_i = \{j:X_j\ge X_i\}.
```

If censoring or recruitment differs by institution, subtype, or treatment
access, then the risk sets carry cohort structure. The model may learn an
ordering that is partly a cohort artifact.

## 4. Sparse Event Gradients

Only uncensored events create numerator terms:

```math
i:\delta_i=1.
```

Highly censored pathology cohorts can produce weak or unstable gradients,
especially when the number of event patients per batch is small.

## 5. Slide Representation Bottleneck

If:

```math
z_i=\sum_j a_{ij}h_{ij},
\qquad
\eta_i=w^\top z_i,
```

then:

```math
\eta_i=\sum_j a_{ij}w^\top h_{ij}.
```

This is interpretable, but it can miss:

```text
multi-region interactions
rare but decisive morphology
separate early-risk and late-risk regions
spatial arrangement after pooling
```

## 6. Attention Collapse

Attention MIL with a Cox loss may focus on patches that separate high-risk and
low-risk patients in the training cohort. If several morphologies are
correlated with risk, attention can collapse onto the easiest correlate rather
than the causal morphology.

## 7. One Score For Multiple Mechanisms

The Cox score imposes:

```math
\eta_i
=
\text{single risk coordinate}.
```

But a slide may express multiple mechanisms:

```text
tumor grade
necrosis
lymphocyte infiltration
stromal invasion
vascular patterns
molecular subtype proxies
```

When these mechanisms affect different time regimes, a scalar ordering is too
coarse.

## Diagnostic Questions

Ask:

1. Does the proportional hazards assumption make biological sense?
2. Are high-risk slides high-risk at every time horizon?
3. Is the model evaluated only by concordance?
4. Does censoring differ by cohort or institution?
5. Does the slide encoder preserve the morphology needed for late-risk effects?

## Design Escape Routes

If Cox is too restrictive:

```text
use time-varying coefficients
predict discrete hazards
predict continuous hazards
model competing risks
separate early and late survival heads
use multimodal pathway-specific survival heads
```

The core tradeoff is:

```text
Cox gives stable ranking under censoring, but compresses risk into one scalar.
```
