# Pretraining Contract Matrix

Let the pretraining contract be

```math
\mathfrak P=(V,A,L,G),
```

with views `V`, relations `A`, loss `L`, and representation geometry `G`.

| Family | Views | Positive relation | Loss | Surviving statistic |
|---|---|---|---|---|
| CTransPath/RetCCL | augmented patches | same instance or cluster | contrastive | local angular geometry |
| HIPT | multi-scale crops | teacher agreement | self-distillation | hierarchical representation |
| UNI/Virchow | pathology patches | self-supervised views/tokens | visual SSL | broad patch geometry |
| Phikon-style MIM | visible and masked tokens | hidden-content prediction | reconstruction | predictable morphology |
| CONCH/PLIP | image and text | paired caption | image-text alignment; exact losses differ | semantic shared space |
| Prov-GigaPath | tiles and slide | slide context | slide-level pretraining | long-context WSI statistic |
| TITAN | slide and report/text | paired slide semantics | multimodal alignment | retrieval-ready slide geometry |
| KEEP | image and knowledge text | knowledge-conditioned pairing | enhanced alignment | knowledge-biased semantic space |
| distillation | teacher and student | teacher function/relation | KD/feature loss | compressed teacher statistic |

Model comparison is incomplete without the relation column. Two models called
self-supervised can preserve different information because their positive pairs
differ.
