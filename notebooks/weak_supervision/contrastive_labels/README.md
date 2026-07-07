# Contrastive Labels

Contrastive supervision does not directly label a slide or patch. It labels
relations:

```math
(a,b)\in\mathcal{P}
\quad
\text{positive pair},
```

and often treats other sampled items as negatives.

## Files

- `01_contrastive_supervision_objects.md`: positives, negatives, views, and relation labels.
- `02_infonce_and_mutual_information_bounds.md`: InfoNCE denominator and sampling assumptions.
- `03_supervised_contrastive_mil.md`: slide-label positives and MIL ambiguity.
- `04_multimodal_and_retrieval_pairs.md`: image-text, retrieval, and memory supervision.
- `05_false_positives_and_sampling_bias.md`: false negatives, class collision, and cohort bias.
- `06_wsi_contrastive_method_cards.md`: SC-MIL, SCL-WC, LACL, RetCCL/CTransPath, CONCH, and PLIP.

## C/R/G/S Placement

```text
G:
    positives can be defined by augmentations, clusters, coordinates, text, or labels

C:
    representation is shaped by pairwise or multi-view relations

R:
    slide, patch, region, or prototype embeddings become contrastive objects

S:
    relation labels rather than direct class labels
```
