# WSI Geometry

This notebook family asks:

```text
How is spatial information encoded?
```

Slide representations ask what kind of object a whole slide is. Pooling asks
what statistic survives aggregation. WSI geometry asks which spatial structure
is available to the context and readout operators.

The repository-wide map is:

```math
\widetilde H_i
=
\mathcal{C}(H_i;G_i,S_i),
\qquad
z_i
=
\mathcal{R}(\widetilde H_i;G_i,S_i),
\qquad
\widehat y_i
=
\mathcal{H}(z_i).
```

This folder studies $G_i$.

## Geometry Object

A tiled whole-slide image gives patch embeddings and patch locations:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
C_i
=
\{c_{ij}\}_{j=1}^{n_i},
\qquad
c_{ij}\in\mathbb{R}^{2}.
```

A geometry is any additional structure derived from, imposed on, or learned over
these locations:

```math
G_i
\in
\{\varnothing,\ C_i,\ \Lambda_i,\ A_i,\ T_i,\ A_{\theta,i}\}.
```

Here:

```text
empty geometry:
    patch order and coordinates are ignored

coordinates:
    each patch has a physical or normalized location

grid:
    patches lie on a regular or sparse lattice

graph:
    edges define local or learned neighborhoods

hierarchy:
    patches compose regions, regions compose slide

learned topology:
    adjacency is inferred from features, coordinates, or task supervision
```

## Folder Map

```text
00_problem_setup.md
    geometry as structure, symmetry, and equivalence relation

no_geometry/
    permutation invariance as geometry erasure
    when ignoring geometry is defensible
    failure modes

coordinate_geometry/
    coordinates as marks
    absolute and relative positional encodings
    physical scale and normalization
    failure modes

grid_geometry/
    lattice objects
    sparse grids and windowed context
    aliasing, tissue masks, and missing patches

graph_geometry/
    edges as geometry
    message passing and hop neighborhoods
    smoothing, oversquashing, and wrong adjacency

hierarchy_geometry/
    parent maps and multiscale structure
    tree, DAG, and scale-indexed geometry
    hierarchical readout and effective weights

learned_topology/
    dynamic adjacency
    feature-space versus physical-space geometry
    identifiability and regularization

unifying_view/
    geometry as equivalence relation
    C/R/G/S placement
    paper placement matrix
    failure-mode matrix
```

## Core Distinction

Geometry is not the same as representation.

```text
representation:
    what mathematical object stores the slide

geometry:
    what spatial relations are allowed to affect context or readout
```

A graph representation stores adjacency directly. A set representation can still
use coordinates if the context operator or readout consumes $C_i$. Conversely, a
graph with bad edges has geometry, but the wrong geometry.

## Anchor Papers

- Patch-GCN: whole-slide patches as a spatial point cloud with graph message
  passing.
- HACT-Net: hierarchical cell-to-tissue graph geometry.
- HIPT: hierarchical multiscale transformer geometry.
- TransMIL: transformer context with positional structure over WSI patches.
- MambaMIL and state-space MIL: sequence geometry induced by an ordering.
- WiKG: dynamic graph representation with learned relation topology.
- PANTHER and prototype methods: morphology geometry, usually not physical
  tissue geometry unless coordinates are added.

## Design Question

Every geometry note should answer:

```text
1. What structure is G?
2. Which symmetries does G preserve or break?
3. How does G enter C?
4. How does G enter R?
5. What spatial statistic can survive?
6. What failure mode follows if G is wrong, missing, or overtrusted?
```
