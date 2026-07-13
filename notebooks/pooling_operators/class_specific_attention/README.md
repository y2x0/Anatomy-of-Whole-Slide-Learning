# Class-Specific Attention

Class-specific attention gives each class its own readout measure.

Instead of one attention distribution:

```math
a_{ij},
```

the model learns:

```math
a_{ij}^{(c)}
```

for each class
```math
c
```
.

The core question:

```text
Does each class need to look at different tissue?
```

## C/R/G/S Placement

```text
G:
    none unless coordinates or regions are supplied outside CLAM

C:
    class-specific scoring plus feature shaping from instance clustering

R:
    one softmax-weighted first moment per class

S:
    slide label plus top-k and bottom-k pseudo-instance constraints
```

Surviving statistic:

```math
z_i^{(c)}
=
\mathbb{E}_{j\sim a_i^{(c)}}[h_{ij}].
```

## Files

- `01_class_conditioned_readout.md`: class-indexed attention measures and logits.
- `02_clam_instance_clustering.md`: CLAM-style top-k and bottom-k instance
  constraints.
- `03_loss_geometry.md`: slide loss plus instance loss as representation shaping.
- `04_failure_modes.md`: pseudo-label errors, competing heads, and heatmap traps.

## Anchor Paper

- Lu et al. "Data Efficient and Weakly Supervised Computational Pathology on
  Whole Slide Images" / CLAM. Nature Biomedical Engineering 2021.
  https://arxiv.org/abs/2004.09666
