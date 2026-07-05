# Minimal Correct Reporting

Every survival modeling note or paper should state the following.

## 1. Observed Label

```math
X_i=\min(T_i,C_i),
\qquad
\delta_i=\mathbf{1}[T_i\le C_i].
```

For competing risks:

```math
\Delta_i\in\{0,1,\ldots,C\}.
```

## 2. Risk Object

State whether the model outputs:

```text
scalar risk
discrete hazards
continuous hazard
survival curve
event-time PMF
competing-risk CIF
```

## 3. Slide Representation

State:

```math
H_i
\xrightarrow{\mathcal{C}}
\widetilde{H}_i
\xrightarrow{\mathcal{R}}
z_i.
```

Include whether geometry is:

```text
none
coordinates
graph
sequence
hierarchy
distribution/prototypes
```

## 4. Loss

State the exact loss:

```text
Cox partial likelihood
discrete hazard likelihood
continuous hazard likelihood
PMF likelihood
ranking loss
IPCW horizon loss
combined objective
```

## 5. Censoring Assumption

State the assumed censoring condition:

```math
T_i\perp C_i\mid z_i
```

or explain deviations.

## 6. Metric

State:

```text
C-index variant
time-dependent AUC definition
Brier/IPCW method
calibration horizon
competing-risk event type
```

## 7. Split Unit

State:

```text
patient-level split
slide-level split
institution-level external validation
```

Patient-level is required when multiple slides per patient exist.

## 8. Baseline Recovery

For Cox, state how:

```math
\widehat{\Lambda}_0(t)
```

is estimated, and whether it uses only training data.

## Dense Checklist

```text
label:
    X, delta, event type

output:
    eta, h, lambda, S, p, F_c

architecture:
    C/R/G/S

loss:
    exact formula

censoring:
    assumption and weighting

metric:
    definition and horizon

validation:
    split unit and leakage controls
```

If a survival result does not specify these, it cannot be compared rigorously.
