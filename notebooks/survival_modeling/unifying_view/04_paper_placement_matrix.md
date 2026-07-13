# Paper Placement Matrix

This file places representative papers in the survival design space.

## Classical / Neural Survival

```text
Cox:
    scalar risk, partial likelihood

DeepSurv:
    neural scalar risk, Cox partial likelihood

Cox-nnet:
    neural Cox score for high-dimensional covariates

Nnet-survival:
    discrete hazards, masked likelihood

DeepHit:
    joint discrete event-time/cause PMF; CIF and survival are derived; likelihood
    plus ranking term, competing risks

Deep Cox Mixtures:
    mixture survival, latent regimes

pycox:
    practical neural Cox/non-proportional survival framework
```

## WSI Survival

```text
Patch-GCN:
    graph slide representation, Cox-style survival head

MCAT:
    pathology-genomics co-attention, survival risk

Pathomic Fusion:
    gated tensor fusion for histology/genomics

PORPOISE:
    pan-cancer multimodal weakly supervised prognosis

SurvPath:
    pathway-token transcriptomics plus histology transformer

C2MIL:
    causal graph MIL for robust/interpretable survival

H2-Surv:
    hyperbolic hierarchy for multimodal survival

E2E-ViT:
    end-to-end WSI transformer survival
```

## Matrix

```text
Paper                 Slide object       Fusion/context      Risk object
-----                 ------------       --------------      -----------
DeepSurv              tabular            MLP                 scalar Cox
Nnet-survival         tabular            MLP                 discrete hazard
DeepHit               tabular            MLP                 PMF/CIF
Patch-GCN             graph WSI          GNN                 scalar risk
MCAT                  WSI + omics        co-attention        scalar risk
Pathomic Fusion       WSI + omics        tensor fusion       scalar risk
PORPOISE              WSI + molecular    multimodal MIL      scalar risk
SurvPath              WSI + pathways     transformer         scalar risk
C2MIL                 graph WSI          causal graph MIL    survival risk
H2-Surv               multimodal hierarchy hyperbolic        survival risk
E2E-ViT               full WSI sequence  end-to-end ViT      survival risk
```

## What The Matrix Exposes

Most WSI survival papers change:

```text
slide representation
context operator
fusion operator
```

but often keep:

```text
    scalar survival risk
    C-index evaluation
```

DeepHit is the useful counterexample in this matrix: its primary prediction is
a discrete joint PMF rather than a scalar Cox score. A survival curve or
cause-specific cumulative incidence function is computed from that PMF, so the
output object and the evaluation target must be named separately.

This is the gap: representation innovation is ahead of risk-object innovation.

## Forward Maps Behind The Matrix

The table is a placement summary, not a claim that every implementation uses the
same hidden layers. The following maps isolate the mathematical contracts that
the cited families share.

### Scalar Cox Family

For Cox and DeepSurv-style models, the slide representation enters one scalar:

```math
z_i
=
\mathcal R(\mathcal C(H_i;G_i)),
\qquad
\eta_i=f_{\theta}(z_i),
\qquad
h_i(t\mid H_i)=h_0(t)\exp(\eta_i).
```

The partial likelihood therefore optimizes the ordering and risk-set contrasts
of eta. The slide encoder is not directly trained to output a calibrated survival
curve unless a baseline hazard estimate and the proportional-hazards assumptions
are also used.

### Discrete Hazards And DeepHit

For a discrete-time hazard head, one logit per interval gives

```math
\widehat h_{ik}=\sigma(a_{ik}),
\qquad
\widehat S_i(\tau_k)
=
\prod_{\ell=1}^{k}(1-\widehat h_{i\ell}).
```

DeepHit-style competing-risk output instead assigns a joint PMF over time and
event type:

```math
\widehat p_{ikc}
=
\frac{\exp(a_{ikc})}
{\sum_{\ell=1}^{K}\sum_{d=1}^{C}\exp(a_{i\ell d})},
\qquad
\widehat F_{ic}(\tau_k)
=
\sum_{\ell\le k}\widehat p_{i\ell c}.
```

This is not a vector of independent hazards. The joint normalization couples all
time-event cells, and the survival function is derived by summing the PMF over
all causes and times up to the horizon.

### Patch-GCN

Patch-GCN makes the WSI object a coordinate-derived graph before the survival
head:

```math
\widetilde H_i
=
\mathrm{GNN}_{\theta}(H_i,A_i),
\qquad
z_i
=
\mathcal R_{\mathrm{graph}}(\widetilde H_i,A_i),
\qquad
\eta_i=w^{\mathsf T}z_i+b.
```

The graph changes the statistic reaching the Cox-style head: a node can carry
information from its graph neighborhood before readout. The survival objective
still constrains the final risk object, not a unique node-level causal map.

### MCAT And Pathomic Fusion

For multimodal WSI models, the fusion operator is part of the representation:

```math
z_i^{\mathrm{path}}
=
\mathcal R_p(\mathcal C_p(H_i)),
\qquad
z_i^{\mathrm{clin}}
=
\mathcal R_c(\mathcal C_c(X_i)).
```

MCAT-style co-attention creates cross-modal summaries before the survival head,
whereas Pathomic Fusion explicitly introduces multiplicative interactions:

```math
z_i
=
\Phi\left(
z_i^{\mathrm{path}},
z_i^{\mathrm{clin}},
z_i^{\mathrm{path}}\otimes z_i^{\mathrm{clin}}
\right),
\qquad
\eta_i=f_{\theta}(z_i).
```

The distinction is important: late concatenation can only use interactions learned
by the head, while tensor fusion exposes a higher-order interaction basis before
the final risk function.

### SurvPath, C2MIL, H2-Surv, And E2E-ViT

These anchors change different axes even when they report a scalar survival risk.
SurvPath inserts pathway-level structure into the multimodal context; C2MIL adds
semantic and topological causal structure to MIL; H2-Surv changes the geometry of
the multimodal latent space through hierarchical hyperbolic representations; and
E2E-ViT moves more of the slide encoder into end-to-end task training. Their
common scalar-head abstraction is

```math
\eta_i
=
f_{\theta}\left(
\mathcal R_{\theta}
\left(
\mathcal C_{\theta}(H_i,X_i;G_i)
\right)
\right),
```

but the meaning of G, C, and R is different in each case. A paper placement is
therefore incomplete unless it names both the risk object and the mechanism that
creates the slide statistic.

## Reporting Consequence

For every row in this matrix, report the tuple

```math
\left(
\text{slide object},
\text{context operator},
\text{readout operator},
\text{risk object},
\text{loss},
\text{metric}
\right).
```

Two methods that share a C-index can still optimize different likelihoods and
predict different mathematical objects. The metric alone does not make them
comparable.
