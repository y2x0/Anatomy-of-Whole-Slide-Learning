# WSI Survival Models

Whole-slide survival models combine a WSI representation with a survival risk
object.

The general map is:

```math
S_i
\to
H_i=\{h_{ij}\}_{j=1}^{n_i}
\xrightarrow{\mathcal{C}}
\widetilde{H}_i
\xrightarrow{\mathcal{R}}
z_i
\xrightarrow{\mathcal{H}_{\operatorname{surv}}}
\text{risk object}.
```

The survival head may output:

```math
\eta_i,\quad
h_{ik},\quad
\lambda_i(t),\quad
S_i(t),\quad
p_{ik},\quad
F_{ic}(t).
```

## Files

- `01_wsi_survival_pipeline.md`: the full WSI-to-risk map.
- `02_mil_survival.md`: mean, attention, prototype, and additive MIL survival.
- `03_graph_transformer_state_space_survival.md`: context-heavy WSI survival.
- `04_end_to_end_and_foundation_features.md`: frozen FMs vs end-to-end models.
- `05_paper_map.md`: Patch-GCN, MCAT, PORPOISE, SurvPath, C2MIL, H2-Surv.
- `06_failure_modes.md`: WSI-specific survival failures.

## Anchor Papers

- Patch-GCN. https://arxiv.org/abs/2107.13048
- MCAT. https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.pdf
- Pathomic Fusion. https://arxiv.org/abs/1912.08937
- PORPOISE. https://arxiv.org/abs/2108.02278
- SurvPath. https://arxiv.org/abs/2304.06819
- C2MIL. https://arxiv.org/abs/2509.20152
