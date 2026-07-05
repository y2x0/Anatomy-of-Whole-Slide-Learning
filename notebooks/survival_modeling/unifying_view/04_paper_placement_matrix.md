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
    discrete PMF, likelihood plus ranking, competing risks

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

This is the gap: representation innovation is ahead of risk-object innovation.
