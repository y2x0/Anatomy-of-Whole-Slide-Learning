# Hierarchy Geometry

Hierarchy geometry represents tissue as nested units:

```text
cell
patch
region
tissue compartment
slide
```

Mathematically, hierarchy is a set of levels plus parent maps:

```math
G_i
=
\left(
\{V_i^{(\ell)}\}_{\ell=0}^{L},
\{\pi_i^{(\ell)}\}_{\ell=0}^{L-1},
\{\Delta_\ell\}_{\ell=0}^{L}
\right).
```

## Files

- `01_parent_maps_and_multiscale_units.md`: hard containment and child sets.
- `02_tree_dag_and_scale_index.md`: tree, DAG, soft assignment, and scale.
- `03_hierarchical_context_and_readout.md`: local-to-global geometry.
- `04_failure_modes.md`: premature compression, bad regions, and scale leakage.

## C/R/G/S Placement

```text
G:
    levels, parent maps, scale index, and cross-level edges

C:
    local context within levels and messages across levels

R:
    region-to-slide or multiscale pooling

S:
    slide labels, region pseudo-labels, self-supervised scale objectives
```
