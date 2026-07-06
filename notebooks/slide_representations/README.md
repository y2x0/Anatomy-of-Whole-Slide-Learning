# Slide Representations

This notebook family asks:

```text
What mathematical object represents a whole slide?
```

The answer is not always "a bag of patches." A whole-slide image can be treated
as a set, sequence, graph, hierarchy, empirical distribution, retrieval query,
or foundation-model latent object.

The representation choice determines which maps are natural.

```math
S_i
\xrightarrow{\operatorname{tile}}
\{x_{ij}\}_{j=1}^{n_i}
\xrightarrow{E}
H_i=\{h_{ij}\}_{j=1}^{n_i}
\xrightarrow{\operatorname{representation}}
\mathcal{X}_i.
```

Then a model learns:

```math
\mathcal{X}_i
\xrightarrow{\mathcal{F}}
z_i
\xrightarrow{\mathcal{H}}
\widehat{y}_i.
```

The representation families are:

```text
set_representations/
    slide = unordered finite set of patch embeddings

sequence_representations/
    slide = ordered sequence of patch embeddings

graph_representations/
    slide = graph of patches, regions, cells, or tissue units

hierarchy_representations/
    slide = nested multiscale tissue object

distribution_representations/
    slide = empirical distribution over morphology

retrieval_memory_representations/
    slide = query plus retrieved memory neighborhood

foundation_latent_representations/
    slide = object in pretrained latent geometry

unifying_view/
    all representation families as choices of structure, context, and readout
```

## Core Distinction

Aggregation asks:

```text
How do instances become a prediction?
```

Representation asks:

```text
What kind of mathematical object is the slide before prediction?
```

This distinction matters because the representation defines the valid symmetry:

```text
set:
    permutation invariance

sequence:
    order-aware recurrence or attention

graph:
    permutation equivariance plus adjacency-aware message passing

hierarchy:
    scale-aware composition

distribution:
    measure-level equivalence

retrieval memory:
    similarity under an external archive

foundation latent:
    pretrained geometry
```

## Reading Order

1. `set_representations/`
2. `sequence_representations/`
3. `graph_representations/`
4. `hierarchy_representations/`
5. `distribution_representations/`
6. `retrieval_memory_representations/`
7. `foundation_latent_representations/`
8. `unifying_view/`

The unifying view then places the families into one C/R/G/S decomposition.

## Anchor Papers

- Zaheer et al. "Deep Sets." NeurIPS 2017.
  https://arxiv.org/abs/1703.06114
- Lee et al. "Set Transformer." ICML 2019.
  https://arxiv.org/abs/1810.00825
- Vinyals et al. "Order Matters: Sequence to sequence for sets." ICLR 2016.
- Yang et al. "MambaMIL: Enhancing Long Sequence Modeling with Sequence
  Reordering in Computational Pathology." MICCAI 2024.
  https://arxiv.org/abs/2403.06800
- Chen et al. "Whole Slide Images are 2D Point Clouds: Context-Aware Survival
  Prediction using Patch-based Graph Convolutional Networks." MICCAI 2021.
  https://arxiv.org/abs/2107.13048
- Pati et al. "Hierarchical Cell-to-Tissue Graph Neural Network for
  Histopathological Image Classification." Medical Image Analysis 2022.
  https://arxiv.org/abs/2007.00584
- Li et al. "Dynamic Graph Representation with Knowledge-aware Attention for
  Histopathology Whole Slide Image Analysis." CVPR 2024.
  https://arxiv.org/html/2403.07719
- Chen et al. "Scaling Vision Transformers to Gigapixel Images via
  Hierarchical Self-Supervised Learning." CVPR 2022.
  https://arxiv.org/abs/2206.02647
- Song et al. "Morphological Prototyping for Unsupervised Slide Representation
  Learning in Computational Pathology." CVPR 2024.
  https://arxiv.org/abs/2405.11643
- Tavolara et al. "SAMPLER: Empirical distribution representations for rapid
  analysis of whole slide tissue images." 2023.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10418159/
- Kalra et al. "Yottixel: An Image Search Engine for Large Archives of
  Histopathology Whole Slide Images." Medical Image Analysis 2020.
  https://arxiv.org/abs/1911.08748
- Chen et al. "Fast and scalable search of whole-slide images via
  self-supervised deep learning." Nature Biomedical Engineering 2022.
  https://www.nature.com/articles/s41551-022-00929-8
- Chen et al. "A General-Purpose Self-Supervised Model for Computational
  Pathology." Nature Medicine 2024.
  https://arxiv.org/abs/2308.15474
- Zimmermann et al. "A foundation model for clinical-grade computational
  pathology and rare cancers detection." Nature Medicine 2024.
  https://www.nature.com/articles/s41591-024-03141-0
- Xu et al. "A whole-slide foundation model for digital pathology from
  real-world data." Nature 2024.
  https://www.nature.com/articles/s41586-024-07441-w
- Lu et al. "Towards A Visual-Language Foundation Model for Computational
  Pathology." Nature Medicine 2024.
  https://arxiv.org/abs/2307.12914
- Shaikovski et al. "PRISM: A Multi-Modal Generative Foundation Model for
  Slide-Level Histopathology." 2024.
  https://arxiv.org/abs/2405.10254
- Ding et al. "Multimodal Whole Slide Foundation Model for Pathology." 2024.
  https://arxiv.org/abs/2411.19666
