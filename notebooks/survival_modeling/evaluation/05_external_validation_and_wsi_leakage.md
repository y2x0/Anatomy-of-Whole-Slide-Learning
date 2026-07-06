# External Validation And WSI Leakage

Survival evaluation in computational pathology has extra failure modes because
slides, patients, institutions, scanners, and molecular cohorts are structured.

## Patient-Level Splits

If multiple slides exist per patient:

```math
S_{i1},S_{i2},\ldots,S_{im_i},
```

all slides from patient $i$ must stay in the same split.

Slide-level splitting leaks patient morphology and outcome.

## Institution Leakage

Let $A_i$ be acquisition site. If:

```math
A_i\not\perp T_i
```

and site is correlated with stain/scanner/style, the model may learn site
surrogates.

External validation should report:

```text
held-out institution
held-out scanner/stain protocol
held-out cancer cohort
temporal validation
```

## Censoring Shift

The censoring distribution:

```math
G(t\mid A)=\Pr(C>t\mid A)
```

can differ by institution. IPCW metrics estimated on pooled data may be unstable
or biased if censoring is strongly site-dependent.

Evaluation should inspect:

```math
\widehat{G}_{\text{train}}(t),
\qquad
\widehat{G}_{\text{test}}(t).
```

## Patch Leakage

If tiles are randomly split rather than slides or patients:

```math
x_{ij}\in\mathrm{train},
\qquad
x_{ik}\in\mathrm{test},
```

then evaluation is invalid. Patch-level feature extractors may memorize stain,
tissue preparation, or patient-specific morphology.

## Feature Extraction Leakage

Foundation model features are usually safe if pretrained externally. But leakage
can enter through:

```text
normalization fitted on all data
prototype dictionaries learned on all data
PCA or clustering fit before splitting
stain normalization templates using test slides
hyperparameter selection on test cohorts
```

Every unsupervised preprocessing step that sees the test set can leak.

## Survival-Specific Leakage

Risk-set losses use other patients in denominators:

```math
\log\sum_{j\in R_i}\exp(\eta_j).
```

Risk sets must be built within training data during model fitting. Test samples
must not enter training risk-set denominators or baseline-hazard estimation
unless explicitly evaluating a transductive method.

## Baseline Hazard Recovery

For Cox models, calibrated survival needs:

```math
\widehat{\Lambda}_0(t).
```

This baseline should be estimated on training data, then applied to validation
or test patients:

```math
\widehat{S}_{\text{test}}(t\mid z_i)
=
\exp[-\exp(\eta_i)\widehat{\Lambda}_{0,\mathrm{train}}(t)].
```

Estimating the baseline on test outcomes leaks label distribution.

## Dense Checklist

```text
split by patient
keep all slides from a patient together
fit preprocessing inside train split
estimate censoring weights from appropriate train data
estimate Cox baseline on train data
report event counts and censoring by split
use external cohorts when possible
avoid patch-level leakage
```

For WSI survival, evaluation quality is often more important than architecture
novelty.
