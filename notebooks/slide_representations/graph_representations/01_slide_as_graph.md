# Slide As Graph

A graph representation defines:

```math
G_i=(V_i,E_i),
\qquad
H_i=\{h_{iv}\}_{v\in V_i}.
```

The full slide object is:

```math
\mathcal{X}_i=(G_i,H_i).
```

If there are edge features:

```math
e_{iuv}\in\mathbb{R}^{r}
```

for $(u,v)\in E_i$, then:

```math
\mathcal{X}_i=(V_i,E_i,H_i,E_i^{\text{feat}}).
```

## Node Choices

Patch graph:

```text
node = image patch
feature = patch embedding
```

Cell graph:

```text
node = nucleus or cell
feature = cell morphology / phenotype
```

Region graph:

```text
node = tissue region or cluster
feature = region embedding
```

Heterogeneous graph:

```text
multiple node and edge types
```

## Edge Choices

Spatial kNN:

```math
(u,v)\in E
\quad\text{if}\quad
v\in\mathrm{kNN}(u).
```

Radius graph:

```math
(u,v)\in E
\quad\text{if}\quad
\|c_u-c_v\|\le r.
```

Similarity graph:

```math
(u,v)\in E
\quad\text{if}\quad
\mathrm{sim}(h_u,h_v)\ge\tau.
```

Learned graph:

```math
A_{uv}=g_\theta(h_u,h_v,c_u,c_v).
```

## Symmetry

A graph model should be invariant to node relabeling.

For permutation matrix $P$:

```math
H'=PH,
\qquad
A'=PAP^\top.
```

A graph-level model should satisfy:

```math
f(H',A')=f(H,A).
```

A node-level representation should be equivariant:

```math
\Phi(H',A')=P\Phi(H,A).
```

## Dense Summary

```math
\mathcal{X}_i=(V_i,E_i,H_i,A_i)
```

represents the slide as entities and relationships. The graph is not just a
model architecture; it is a claim about which tissue units interact.
