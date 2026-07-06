# Multimodal Survival Failure Modes

Multimodal survival models can improve prediction, but they can also obscure the
source of risk.

## 1. Dominant Modality Collapse

If one modality is much more predictive:

```math
\Phi(z^p,z^g)\approx \Phi(z^g),
```

then pathology contributes little even if the model is described as multimodal.

Check by modality ablation:

```math
\Delta_{\text{path}}
=
\operatorname{Perf}(p,g)-\operatorname{Perf}(g).
```

## 2. Missingness Confounding

If genomic data are available only for selected patients:

```math
M^g\not\perp T,
```

then the missingness pattern itself may predict survival.

## 3. Cross-Modal Leakage

If molecular features are measured after treatment or diagnosis workflows that
depend on prognosis, they may encode downstream information not available at
prediction time.

## 4. Interpretability Overclaim

Attention between a gene/pathway token and a patch token:

```math
A_{rj}
```

does not prove that pathway $r$ causes morphology $j$, or vice versa.

It is an interaction used for prediction.

## 5. Scale Mismatch

Pathology is spatial and local:

```math
\{h_{ij}\}_{j=1}^{n_i}.
```

Bulk transcriptomics is global:

```math
g_i.
```

Fusion may align local morphology to a global mixture signal, which can blur
cell-type-specific mechanisms.

## 6. Small Paired Cohorts

The paired multimodal set may be much smaller than the pathology-only set:

```math
n_{\text{paired}}\ll n_{\text{path}}.
```

Large fusion models can overfit.

## Dense Checklist

```text
compare unimodal baselines
ablate each modality
report paired cohort size
model missingness explicitly
avoid causal claims from attention alone
validate on external cohorts
check calibration, not only C-index
```

Multimodal survival should prove that fusion improves the risk object, not just
that more data streams were concatenated.
