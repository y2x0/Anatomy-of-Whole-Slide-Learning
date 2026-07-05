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

The first three representation families are:

```text
set_representations/
    slide = unordered finite set of patch embeddings

sequence_representations/
    slide = ordered sequence of patch embeddings

graph_representations/
    slide = graph of patches, regions, cells, or tissue units
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
```

## Reading Order

1. `set_representations/`
2. `sequence_representations/`
3. `graph_representations/`

These three cover the most common ways to turn patch embeddings into a slide
object before task-specific aggregation.

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
- Li et al. "Dynamic Graph Representation with Knowledge-aware Attention for
  Histopathology Whole Slide Image Analysis." CVPR 2024.
  https://arxiv.org/html/2403.07719
