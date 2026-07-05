# Multimodal Survival

Multimodal survival combines pathology with clinical, genomic, transcriptomic,
or report-derived information.

Let:

```math
z_i^{(1)},z_i^{(2)},\ldots,z_i^{(M)}
```

be modality representations. Fusion builds:

```math
z_i=\Phi(z_i^{(1)},\ldots,z_i^{(M)}),
```

then:

```math
z_i\mapsto \text{survival risk object}.
```

## Files

- `01_fusion_operators.md`: early, late, tensor, attention, and product fusion.
- `02_pathology_genomics_tensor_fusion.md`: Pathomic Fusion/PORPOISE math.
- `03_coattention_and_pathway_tokens.md`: MCAT and SurvPath-style fusion.
- `04_missing_modalities.md`: missingness and modality dropout.
- `05_hierarchy_hyperbolic_recent_models.md`: hierarchy and hyperbolic models.
- `06_failure_modes.md`: multimodal-specific traps.
