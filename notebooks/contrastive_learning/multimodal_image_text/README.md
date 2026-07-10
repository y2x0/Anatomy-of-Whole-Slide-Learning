# Multimodal Image-Text Contrast

This portion asks:

```text
What does image-text alignment identify when pathology pairs are weak,
many-to-many, or partly generated?
```

The notes derive the exact objectives used by CLIP, PLIP, CONCH, and CPLIP,
then separate objective geometry from pair construction and WSI readout.
PathCLIP enters as a robustness study rather than a new loss family.

## Notes

```text
01_paired_multimodal_statistical_object.md
    observed pairs, latent semantic compatibility, and correspondence noise

02_clip_symmetric_objective_and_gradients.md
    exact bidirectional loss, matrix gradients, and learned logit scale

03_bidirectional_density_ratios_and_nonidentifiability.md
    what each softmax estimates and what alignment cannot identify

04_zero_shot_text_prototypes_and_prompt_geometry.md
    text embeddings as normalized classifier weights

05_plip_openpath_pair_distribution.md
    social-media, reply, and retrieval-derived pair mechanisms

06_plip_false_negatives_and_retrieval_selection_bias.md
    one-to-many semantics, imported geometry, and distribution shift

07_conch_dual_readouts_and_multimodal_decoder.md
    one global contrast token, 256 caption tokens, and tensor shapes

08_conch_contrastive_captioning_objective.md
    exact joint loss, information paths, and failure modes

09_conch_wsi_prompt_conditioned_topk_readout.md
    MI-Zero tile scores and class-specific surviving statistics

10_cplip_bag_construction_map.md
    dictionary assignment, generated text, retrieval, and pruning

11_cplip_mil_nce_objective_and_gradients.md
    exact many-to-many log-sum-exp loss and bag-size effects

12_pathclip_corruption_geometry.md
    robustness of zero-shot classification and retrieval

13_crgs_multimodal_alignment_and_failure_matrix.md
    unified placement and design consequences
```

## Primary Paper Boundary

Every paper used here is already listed in the private research map. The core
anchors are CLIP, PLIP/OpenPath, CONCH, PathCLIP, and CPLIP. No paper PDF or
private research artifact is copied into this public folder.
