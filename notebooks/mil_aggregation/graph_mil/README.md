# Graph MIL

This portion asks a narrower question than the general graph-learning family:

```text
How does a graph-contextualized patch field become one slide prediction?
```

A graph MIL model is not specified by the readout alone. It is a composition:

```math
H_i^{(0)}
\xrightarrow{\;\mathcal{G}_i\;}
\text{graph support and attributes}
\xrightarrow{\;\mathcal{C}_{\theta}\;}
\widetilde H_i
\xrightarrow{\;\mathcal{R}_{\theta}\;}
z_i
\xrightarrow{\;\mathcal{H}_{\theta}\;}
\widehat y_i.
```

The graph changes the statistic available to the readout. A mean or attention
readout over original patches is not equivalent to the same readout over nodes
whose states have exchanged messages.

## Notes

```text
01_graph_as_mil_object_and_equivariance.md
    graph input, permutation equivariance, and the set/graph boundary

02_patch_gcn_coordinate_context_and_survival.md
    coordinate kNN support, DeepGCN-style context, attention readout, and Cox

03_heat_typed_context_and_pseudo_label_readout.md
    HEAT node/edge heterogeneity and PL-Pool as a typed graph readout

04_wikg_dynamic_support_and_knowledge_readout.md
    learned directed top-k support, knowledge attention, and mean/max readout

05_hact_hierarchy_as_graph_mil.md
    cell-to-tissue assignment, hierarchical context, and tissue-level pooling

06_graph_readouts_and_surviving_statistics.md
    mean, max, attention, typed, hierarchical, and task-head statistics

07_graph_mil_failure_modes_and_counterexamples.md
    oversmoothing, oversquashing, topology errors, sum confounding, and toy
    collisions

08_graph_mil_crgs_design_matrix.md
    unified C/R/G/S placement, paper-faithful comparison, and design rules
```

## Source Boundary

The paper-specific derivations use mapped sources already present in the
private research map and the graph-learning family:

- Gilmer et al., *Neural Message Passing for Quantum Chemistry*.
  https://arxiv.org/abs/1704.01212
- Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional
  Networks*. https://arxiv.org/abs/1609.02907
- Velickovic et al., *Graph Attention Networks*.
  https://arxiv.org/abs/1710.10903
- Chen et al., *Whole Slide Images are 2D Point Clouds: Context-Aware Survival
  Prediction using Patch-based Graph Convolutional Networks*.
  https://arxiv.org/abs/2107.13048
- Pati et al., *HACT-Net: A Hierarchical Cell-to-Tissue Graph Neural Network
  for Histopathological Image Classification*.
  https://arxiv.org/abs/2007.00584
- Pati et al., *Hierarchical graph representations in digital pathology*.
  https://arxiv.org/abs/2102.11057
- Chan et al., *Histopathology Whole Slide Image Analysis With Heterogeneous
  Graph Representation Learning*. https://arxiv.org/abs/2307.04189
- Li et al., *Dynamic Graph Representation with Knowledge-aware Attention for
  Histopathology Whole Slide Image Analysis*.
  https://arxiv.org/abs/2403.07719

These notes separate claims stated by the papers from dimension-safe notation
used to expose the operator. They do not silently turn a graph construction
into an anatomical truth.
