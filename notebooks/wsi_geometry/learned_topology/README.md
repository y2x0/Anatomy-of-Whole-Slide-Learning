# Learned Topology

Learned topology makes geometry a model output.

Instead of fixing adjacency from coordinates:

```math
A_i
=
A(C_i),
```

the model learns:

```math
A_{\theta,i}
=
A_\theta(H_i,C_i,S_i).
```

WiKG-style dynamic graph models fit this family: the relevant relations are not
only physical adjacency but learned task-dependent connectivity.

## Files

- `01_dynamic_adjacency.md`: learned edges and differentiable topology.
- `02_feature_space_vs_physical_space.md`: morphology relations versus tissue relations.
- `03_identifiability_regularization_and_failure_modes.md`: shortcut topology and constraints.

## C/R/G/S Placement

```text
G:
    learned adjacency or relation tensor

C:
    message passing over learned neighborhoods

R:
    pooling after dynamic contextualization

S:
    labels and regularizers shape topology
```
