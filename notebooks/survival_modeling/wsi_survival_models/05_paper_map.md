# Paper Map

This note places representative WSI survival papers by mathematical role.

## Patch-GCN

Core object:

```math
\text{slide} = \text{spatial graph of patches}.
```

Mathematical map:

```math
H_i,G_i
\xrightarrow{\text{GNN}}
\widetilde{H}_i
\xrightarrow{\text{attention/readout}}
z_i
\xrightarrow{\mathrm{survival\ head}}
\eta_i.
```

Inductive bias:

```text
prognostic morphology depends on spatially contextual patch interactions
```

Relevant source: Patch-GCN.
https://arxiv.org/abs/2107.13048

## MCAT

Core object:

```math
\text{slide patches} \leftrightarrow \text{omic tokens}.
```

Co-attention maps histology and genomics into a fused survival representation.

Mathematical map:

```math
H_i^{\text{path}},H_i^{\text{omic}}
\xrightarrow{\text{coattention}}
Z_i
\xrightarrow{\text{fusion}}
z_i
\xrightarrow{\text{survival}}
\eta_i.
```

Inductive bias:

```text
prognosis depends on cross-modal interactions
```

Relevant source: MCAT.
https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.pdf

## Pathomic Fusion

Core operator:

```math
z_{\text{fused}}
=
\operatorname{TensorFusion}
(z_{\text{path}},z_{\text{omic}}).
```

Tensor fusion models pairwise and higher-order interactions between modalities.

Inductive bias:

```text
histology and genomics interact multiplicatively
```

Relevant source: Pathomic Fusion.
https://arxiv.org/abs/1912.08937

## PORPOISE

Core object:

```math
\text{pan-cancer histology-genomic survival model}.
```

PORPOISE uses weakly supervised multimodal deep learning to connect local
morphology and global molecular features across cancer types.

Mathematical role:

```text
multimodal survival with interpretable local/global feature importance
```

Relevant source: PORPOISE.
https://arxiv.org/abs/2108.02278

## SurvPath

Core object:

```math
\text{histology patch tokens} \times \text{biological pathway tokens}.
```

Mathematical map:

```math
H_i^{\text{path}},
P_i^{\text{pathway}}
\xrightarrow{\mathrm{multimodal\ transformer}}
z_i
\xrightarrow{\text{survival}}
\eta_i.
```

Inductive bias:

```text
transcriptomics should be tokenized into biological pathways before fusion
```

Relevant source: SurvPath.
https://arxiv.org/abs/2304.06819

## C2MIL

Core object:

```math
\text{graph MIL with semantic and topological causal filtering}.
```

Mathematical role:

```text
remove semantic confounding and non-causal topology before survival readout
```

Relevant source: C2MIL.
https://arxiv.org/abs/2509.20152

## H2-Surv

Core object:

```math
\text{hierarchical multimodal representation in hyperbolic space}.
```

Mathematical role:

```text
use non-Euclidean geometry for hierarchical pathology/omics survival structure
```

Relevant source: H2-Surv.
https://openaccess.thecvf.com/content/CVPR2026/html/Yang_H2-Surv_Hierarchical_Hyperbolic_Multimodal_Representation_Learning_for_Survival_Prediction_CVPR_2026_paper.html

## E2E-ViT

Core object:

```math
\text{end-to-end WSI transformer survival model}.
```

Mathematical role:

```text
survival gradients update WSI-scale visual representation rather than only a
post-hoc MIL head
```

Relevant source: E2E-ViT.
https://openaccess.thecvf.com/content/CVPR2026/html/Li_Turning_Pre-Trained_Vision_Transformers_into_End-to-End_Histopathology_Whole_Slide_Image_CVPR_2026_paper.html

## Dense Placement Matrix

```text
Patch-GCN:
    graph context, scalar survival head

MCAT:
    cross-attention fusion, scalar survival head

Pathomic Fusion:
    gated tensor fusion, prognosis/diagnosis

PORPOISE:
    pan-cancer weak multimodal prognosis

SurvPath:
    pathway-token multimodal transformer

C2MIL:
    causal graph MIL for robust survival

H2-Surv:
    hyperbolic hierarchy for multimodal survival

E2E-ViT:
    end-to-end WSI survival representation learning
```
