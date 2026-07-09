# Problem Setup

Graph learning is not the same as graph representation.

A graph representation says:

```text
the slide is nodes plus edges
```

A graph learning operator says:

```text
node states are transformed by messages sent along those edges
```

For a WSI graph:

```math
G_i
=
(V_i,E_i),
\qquad
H_i^{(0)}
=
\{h_{iv}^{(0)}\}_{v\in V_i},
```

where nodes can be patches, cells, glands, regions, or slide-level entities.

This first portion uses a row-vector convention:

```math
H_{iv:}^{(\ell)}
=
h_{iv}^{(\ell)}.
```

Thus matrix formulas use:

```math
H^{(\ell)}W^{(\ell)},
```

and nodewise formulas write the same projection as:

```math
h_v^{(\ell)}W^{(\ell)}.
```

## C/R/G/S Placement

Graph learning most often lives in the context operator:

```math
\widetilde H_i
=
\mathcal{C}_{\theta}
\left(
H_i^{(0)};G_i
\right).
```

Then readout maps contextualized node states to a slide statistic:

```math
z_i
=
\mathcal{R}_{\theta}
\left(
\widetilde H_i;G_i
\right).
```

For the vanilla GCN, GraphSAGE, and GAT operators in this first portion,
supervision does not enter the inference-time graph operator. It enters through
the training objective:

```math
\mathcal{L}
\left(
\mathcal{H}_{\theta}(z_i),
S_i
\right).
```

The task head is:

```math
\widehat y_i
=
\mathcal{H}_{\theta}(z_i).
```

In survival:

```math
\eta_i
=
w^\top z_i
```

or:

```math
\widehat q_i
=
\mathcal{H}_{\theta}(z_i)
```

for discrete hazards.

## Generic Message Passing

Let the allowed source set for node `v` be:

```math
\mathcal{A}(v)
=
\{u:(u,v)\in E_i\}.
```

Equivalently, in adjacency-matrix notation this portion uses a row-target,
column-source convention:

```math
A_{vu}=1
\quad
\Longleftrightarrow
\quad
u\to v.
```

A message-passing layer has three pieces:

```math
m_{ivu}^{(\ell)}
=
\psi_{\theta}^{(\ell)}
\left(
h_{iv}^{(\ell)},
h_{iu}^{(\ell)},
e_{ivu}
\right),
```

```math
\bar m_{iv}^{(\ell)}
=
\rho
\left(
\{m_{ivu}^{(\ell)}:u\in\mathcal{A}(v)\}
\right),
```

```math
h_{iv}^{(\ell+1)}
=
\phi_{\theta}^{(\ell)}
\left(
h_{iv}^{(\ell)},
\bar m_{iv}^{(\ell)}
\right).
```

Here:

```text
psi:
    message map

rho:
    permutation-invariant neighborhood aggregation

phi:
    node update

e_ivu:
    optional edge feature, distance, relation type, or geometry
```

## What Moves?

Graph learning moves representations, not pixels:

```math
h_{iu}^{(\ell)}
\longrightarrow
m_{ivu}^{(\ell)}
\longrightarrow
h_{iv}^{(\ell+1)}.
```

The edge controls whether movement is allowed:

```math
(u,v)\notin E_i
\quad
\Longrightarrow
\quad
m_{ivu}^{(\ell)}
\text{ is absent.}
```

The message function controls what is sent. The aggregator controls what
survives after many neighbors send messages to one node.

## Receptive Field

After one layer, node `v` depends on one-hop neighbors:

```math
h_{iv}^{(1)}
=
F_{\theta}^{(1)}
\left(
G_i[\mathcal{B}_1(v)],
\{h_{iu}^{(0)}:u\in\mathcal{B}_1(v)\},
\{e_{iur}:(u,r)\in E_i[\mathcal{B}_1(v)]\}
\right).
```

After `L` layers:

```math
h_{iv}^{(L)}
=
F_{\theta}^{(L)}
\left(
G_i[\mathcal{B}_L(v)],
\{h_{iu}^{(0)}:u\in\mathcal{B}_L(v)\},
\{e_{iur}:(u,r)\in E_i[\mathcal{B}_L(v)]\}
\right).
```

where:

```math
\mathcal{B}_L(v)
=
\{u:d_{G_i}(u,v)\le L\}.
```

Depth expands graph context only along available paths.

## WSI Consequence

In WSI graph modeling, a graph is built from tissue units and their relations.
The context operator is roughly:

```math
(H_i^{(0)},C_i)
\longrightarrow
G_i
\longrightarrow
H_i^{(L)}
\longrightarrow
z_i
\longrightarrow
\eta_i.
```

The method is not just "MIL with a graph." It changes the object that reaches
pooling:

```text
MIL:
    pooled isolated patch embeddings

graph learning:
    pooled neighborhood-contextualized patch embeddings
```

## Dense Summary

Graph learning asks:

```text
what node information is transmitted, along which edges, with what aggregation,
and what contextualized statistic reaches readout?
```

The edge set is a hard inductive bias. A graph learner can reweight, smooth,
sample, or attend along edges, but it cannot use a missing edge unless the
topology is changed by another operator.
