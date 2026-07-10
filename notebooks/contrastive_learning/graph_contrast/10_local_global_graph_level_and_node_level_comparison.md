# Local-Global, Graph-Level, And Node-Level Contrast

## Three Population Experiments

DGI discriminates pair laws:

```math
p(h,s)
\quad\text{versus}\quad
q_{\mathcal{C}}(\widetilde h,s).
```

GraphCL identifies the matching graph view:

```math
p
\left(
\widehat G^{(2)}
\mid
\widehat G^{(1)}
\right)
\quad\text{against batch graph proposals}.
```

GRACE identifies the corresponding node:

```math
p
\left(
v_i
\mid
u_i,
\{v_k\},
\{u_k:k\ne i\}
\right).
```

## Unit Of Representation

| Method | Anchor | Positive | Negatives | Surviving unit |
|---|---|---|---|---|
| DGI | local node/patch embedding | summary of uncorrupted graph | corrupted local embeddings paired with positive summary | node embedding conditioned by one global vector |
| GraphCL | augmented graph embedding | second view of same graph | views of other graphs | one graph-level vector |
| GRACE | node in view one | same node in view two | other nodes in both views | node-level vector |

## Context Radius

For `L` message-passing layers:

```math
h_i
=
F
\left(
G[i,L]
\right).
```

DGI compares this local radius with a global readout. GRACE compares two
corrupted versions of the same radius. GraphCL pools all encoded nodes before
comparison.

## Symmetry

DGI readout is permutation invariant but its local outputs are equivariant:

```math
\mathcal{E}(PX,PAP^{\top})
=
P\mathcal{E}(X,A),
```

```math
\mathcal{R}(PH)
=
\mathcal{R}(H).
```

GraphCL's graph output is invariant. GRACE retains node correspondence and is
equivariant only when both views share the same node relabeling.

## Failure Transfer

Mean-readout blindness affects DGI's summary and any mean-pooled GraphCL
encoder:

```math
\text{same first moment}
\Longrightarrow
\text{same global conditioning or graph vector}.
```

Node-identity false negatives particularly affect GRACE:

```math
\text{exchangeable nodes}
\Longrightarrow
\text{contradictory candidate labels}.
```

Destructive graph augmentations affect GraphCL and GRACE:

```math
\text{task information removed from view}
\Longrightarrow
\text{unrecoverable after encoding}.
```

Corruption shortcuts particularly affect DGI:

```math
\text{easy artifact distinguishes }P,Q_{\mathcal{C}}
\Longrightarrow
\text{high objective without desired semantics}.
```

## They Are Not Ordered By Strength

There is no general implication:

```math
\text{good node contrast}
\Longrightarrow
\text{good graph representation},
```

nor:

```math
\text{good graph contrast}
\Longrightarrow
\text{good node localization}.
```

The result depends on readout sufficiency, augmentation validity, and whether
the downstream target lives at node, region, slide, or patient scale.
