# Graph Representation Failure Modes

Graph representations introduce spatial or relational structure, but the graph
can be wrong.

## 1. Topology Error

If edges do not correspond to meaningful tissue relationships, message passing
propagates irrelevant information.

Example:

```math
(u,v)\in E
```

because of kNN in coordinates, but a tissue gap or artifact separates the
regions.

## 2. Oversmoothing

Repeated message passing can make node states similar:

```math
h_u^{(L)}\approx h_v^{(L)}
```

for many nodes. This erases local morphology.

Oversmoothing is especially problematic when rare local structures drive the
label.

## 3. Oversquashing

Long-range information may need to pass through narrow graph bottlenecks.

If many distant nodes influence one region through few edges, their information
is compressed:

```math
\{h_u:u\in B\}
\to
h_v^{(L)}
```

through limited-dimensional messages.

## 4. Degree Bias

High-degree nodes contribute to many messages and may dominate the representation.

In tissue graphs, degree can correlate with:

```text
tissue density
segmentation artifacts
cellularity
patch sampling density
```

not necessarily biological importance.

## 5. Readout Bottleneck

Even if the graph encodes rich topology:

```math
z=\frac{1}{|V|}\sum_vh_v^{(L)}
```

may discard graph-specific structure.

## 6. Learned Topology Instability

Dynamic graphs:

```math
A_{uv}=g_\theta(h_u,h_v)
```

can overfit, especially with weak slide-level labels.

The learned graph may reflect label shortcuts rather than tissue structure.

## Diagnostic Questions

1. What are nodes?
2. What do edges mean biologically?
3. Does graph depth match the needed spatial range?
4. Is readout preserving topology or only contextualized averages?
5. Is performance robust to graph construction choices?

## Dense Summary

Graph representations replace permutation-invariant bags with relational tissue
objects:

```math
(H,A)
```

but the adjacency matrix is an inductive bias. If
```math
A
```
is wrong, the model's
context is wrong.
