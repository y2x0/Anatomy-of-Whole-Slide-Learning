# WSI Graph Models

This folder derives WSI-specific graph learning papers one paper at a time.

The first anchor is Patch-GCN:

```text
Chen et al.
Whole Slide Images are 2D Point Clouds:
Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks
MICCAI 2021
https://arxiv.org/abs/2107.13048
```

Patch-GCN is useful for this taxonomy because it is not merely "use a GCN on
patches." It specifies a full WSI graph pipeline:

```text
patch extraction
fixed encoder features
coordinate kNN graph
DeepGCN-style local message passing
dense contextual node states
global attention pooling
Cox-style survival supervision
```

## Files

- `01_patch_gcn_slide_object.md`: the patient/slide graph object.
- `02_patch_graph_construction.md`: patch coordinates, node features, and kNN
  support.
- `03_patch_gcn_context_operator.md`: ReLU plus softmax message passing,
  residual mappings, dense connections, and 4-hop context.
- `04_patch_gcn_readout_and_survival.md`: global attention pooling and survival
  risk supervision.
- `05_patch_gcn_failure_modes.md`: support, smoothing, attention, and survival
  bottlenecks.

## C/R/G/S Placement

```text
G:
    coordinate-derived WSI patch graph

C:
    Patch-GCN graph context operator

R:
    global attention pooling over contextualized patch nodes

S:
    patient-level survival labels used through the training objective
```

