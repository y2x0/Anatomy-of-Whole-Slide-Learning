# WSI Survival Failure Modes

WSI survival has all standard survival issues plus slide-specific representation
issues.

## 1. Representation Bottleneck

If:

```math
z_i=\mathcal{R}(H_i)
```

discards rare prognostic morphology, no survival head can recover it.

This is independent of whether the head is Cox, discrete hazard, or continuous
hazard.

## 2. Censoring Correlated With Site

If:

```math
C_i\not\perp A_i
```

where
```math
A_i
```
is institution/scanner/cohort, then the survival loss may learn
site-related features through censoring patterns.

## 3. Weak Event Counts

Survival cohorts often have:

```math
\sum_i\delta_i\ll n.
```

High-capacity WSI encoders can overfit event patients.

## 4. Attention Misinterpretation

Attention weights:

```math
a_{ij}
```

are not automatically causal explanations. A patch can receive high attention
because it is predictive, confounded, or simply useful for cohort separation.

## 5. Time Collapse

Cox heads collapse all prognosis into:

```math
\eta_i.
```

This can hide different early and late morphologic mechanisms.

## 6. Graph Oversmoothing

Graph survival can wash out sharp local risk features:

```math
h_{ij}^{(L)}\approx h_{ik}^{(L)}
```

after too much message passing.

## 7. Foundation Feature Mismatch

Foundation model embeddings may be optimized for diagnosis, retrieval, or
captioning, not survival. Prognostic morphology can be weakly represented.

## Dense Checklist

```text
check event counts
check censoring by cohort
evaluate external validation
report calibration, not only C-index
separate patient and slide splits
test time-specific performance
inspect whether attention is stable
avoid interpreting correlation as mechanism
```

The model's mathematical output must match the clinical claim.
