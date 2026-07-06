# Graph Construction

Graph representation quality depends on how the graph is built.

The graph is not neutral. It defines possible information flow.

## Coordinate kNN Graph

Given patch coordinates $c_v\in\mathbb{R}^{2}$:

```math
\mathcal{N}(v)
=
\operatorname{kNN}_{k}(c_v;\{c_u\}_{u\in V}).
```

Edges:

```math
(u,v)\in E
\quad\text{if}\quad
u\in\mathcal{N}(v).
```

Inductive bias:

```text
nearby patches are relevant context
```

## Radius Graph

```math
(u,v)\in E
\quad\Longleftrightarrow\quad
\|c_u-c_v\|\le r.
```

This preserves physical scale better than fixed kNN, but graph degree varies
with tissue density and missing tissue.

## Morphology Similarity Graph

```math
(u,v)\in E
\quad\Longleftrightarrow\quad
\operatorname{sim}(h_u,h_v)\ge\tau.
```

Inductive bias:

```text
morphologically similar regions should exchange information
```

This can connect distant regions and ignore spatial adjacency.

## Learned Dynamic Graph

A learned adjacency:

```math
A_{uv}
=
\operatorname{softmax}_{u}
g_\theta(h_v,h_u,c_v,c_u).
```

This makes topology model-dependent.

WiKG-style dynamic graph approaches fit here: the graph is not fixed physical
adjacency but learned relation structure.

Mathematically, this changes the slide object from:

```math
(H,C,A_{\text{physical}})
```

to:

```math
(H,C,A_\theta(H,C)).
```

The representation is now end-to-end learned. This can connect distant but
related morphologies, but it also means the graph may encode label shortcuts.

## Heterogeneous Graph

A heterogeneous graph has node types:

```math
\tau(v)\in\mathcal{T}_V
```

and edge types:

```math
\rho(u,v)\in\mathcal{T}_E.
```

Example:

```text
cell node
tissue-region node
patch node
cell-in-region edge
region-adjacent-region edge
cell-near-cell edge
```

HACT-style models use hierarchy between cell and tissue graphs.

More explicitly, a hierarchical cell-to-tissue graph can be written:

```math
V
=
V_{\text{cell}}
\cup
V_{\text{tissue}},
```

```math
E
=
E_{\text{cell-cell}}
\cup
E_{\text{tissue-tissue}}
\cup
E_{\text{cell-tissue}}.
```

The cross-level edge:

```math
(c,r)\in E_{\text{cell-tissue}}
```

means cell $c$ belongs to or is spatially contained in tissue region $r$. This
makes the graph a multiscale tissue object rather than a flat patch adjacency
graph.

## Dense Summary

```text
kNN graph:
    local coordinate context

radius graph:
    physical distance threshold

similarity graph:
    morphology relation

learned graph:
    task-dependent topology

heterogeneous graph:
    typed tissue entities and relations
```

Graph construction is part of the representation, not preprocessing trivia.
