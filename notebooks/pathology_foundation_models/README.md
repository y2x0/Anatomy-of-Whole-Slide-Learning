# Pathology Foundation Models

This notebook family asks:

```text
How was the pathology representation learned before the downstream task?
```

The important object is not the model brand. It is the pretraining statistical
contract: which views are declared equivalent, which pairs are pulled together,
which information is reconstructed, which language concepts are aligned, and
whether a whole slide is represented as a bag, hierarchy, or retrieved memory.

## Portions

1. `01_pretraining_objects_and_contracts.md`
2. `02_contrastive_instance_geometry.md`
3. `03_self_distillation_and_teacher_student_geometry.md`
4. `04_masked_modeling_and_reconstruction.md`
5. `05_pathology_vision_language_alignment.md`
6. `06_slide_level_hierarchy_and_long_context.md`
7. `07_multimodal_knowledge_and_report_alignment.md`
8. `08_retrieval_memory_and_nearest_neighbor_semantics.md`
9. `09_knowledge_distillation_and_model_compression.md`
10. `10_transfer_geometry_and_foundation_model_evaluation.md`
11. `11_crgs_foundation_model_placement.md`

## Primary Sources

- [CTransPath](https://pubmed.ncbi.nlm.nih.gov/35952419/)
- [RetCCL](https://pubmed.ncbi.nlm.nih.gov/36270093/)
- [HIPT](https://arxiv.org/abs/2206.02647)
- [CONCH](https://arxiv.org/abs/2307.12914)
- [UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/)
- [Virchow](https://arxiv.org/abs/2309.07778)
- [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w)
- [TITAN](https://www.nature.com/articles/s41591-025-03982-3)
- [KEEP](https://arxiv.org/html/2412.13126v1)
- [Unified Knowledge Distillation](https://arxiv.org/html/2407.18449v3)
- [Phikon-v2](https://arxiv.org/abs/2409.09173)
