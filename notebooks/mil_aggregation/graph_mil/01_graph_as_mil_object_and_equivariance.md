# Graph As An MIL Object And Equivariance

## 1. From A Bag To A Graph

For slide `i`, let the patch encoder produce a node matrix:

```math
H_i^{(0)}
=
\begin{bmatrix}
h_{i1}^{(0)}\\
\vdots\\
h_{in_i}^{(0)}
\end{bmatrix}
\in
\mathbb{R}^{n_i\times d}.
```

Set MIL exposes only the multiset of rows:

```math
\mathcal{B}_i
=
\left\{
h_{iv}^{(0)}:v\in V_i
\right\}.
```

Graph MIL adds a relation object:

```math
\mathcal{G}_i
=
\left(
V_i,E_i,A_i,X_i,\Xi_i
\right),
```

where `A_i` is adjacency, `X_i` stores node features, and `Xi_i` denotes
optional coordinates, node types, edge features, or assignment maps. The
nodes may still be patches, but the bag is no longer represented by features
alone.

The predictor has the form:

```math
\widehat y_i
=
\mathcal{H}_{\theta}
\left(
\mathcal{R}_{\theta}
\left(
\mathcal{C}_{\theta}
\left(
H_i^{(0)},\mathcal{G}_i
\right)
\right)
\right).
```

The graph is context when it changes node states before readout. It is also
geometry when its support encodes coordinates, feature similarity, hierarchy,
or a learned relation.

## 2. The Correct Symmetry

Node ordering is arbitrary. Let `P_i` be an `n_i` by `n_i` permutation matrix.
Under a reindexing:

```math
H_i^{(0)\prime}
=
P_iH_i^{(0)},
\qquad
A_i^{\prime}
=
P_iA_iP_i^{\top}.
```

A graph context operator should be permutation equivariant:

```math
\mathcal{C}_{\theta}
\left(
P_iH_i^{(0)},P_iA_iP_i^{\top}
\right)
=
P_i\mathcal{C}_{\theta}
\left(
H_i^{(0)},A_i
\right).
```

The graph readout should then be invariant:

```math
\mathcal{R}_{\theta}(P_i\widetilde H_i)
=
\mathcal{R}_{\theta}(\widetilde H_i).
```

Their composition is a well-defined slide function:

```math
f_{\theta}
\left(P_iH_i^{(0)},P_iA_iP_i^{\top}\right)
=
f_{\theta}(H_i^{(0)},A_i).
```

This is not the same as saying the graph is permutation invariant. Its node
order is arbitrary, but its relation structure is part of the input.

## 3. Message Passing Proof Sketch

For a generic message-passing layer:

```math
m_{iv}^{(\ell)}
=
\rho_{\ell}
\left(
\left\{
\phi_{\ell}
\left(
h_{iv}^{(\ell)},h_{iu}^{(\ell)},e_{iuv}
\right)
:u\in\mathcal{N}_i(v)
\right\}
\right),
```

```math
h_{iv}^{(\ell+1)}
=
\zeta_{\ell}
\left(
h_{iv}^{(\ell)},m_{iv}^{(\ell)}
\right).
```

If `rho` is permutation invariant over the neighbor multiset, reindexing the
nodes only reindexes the set of messages reaching each reindexed target. Thus
each target state is carried to its corresponding target state. This is the
local reason that sum, mean, max, and attention normalized over a neighbor set
can be equivariant.

The graph readout must remove the remaining node index. A sum readout is:

```math
z_i^{\mathrm{sum}}
=
\sum_{v\in V_i}\widetilde h_{iv},
```

and an attention readout is:

```math
z_i^{\mathrm{attn}}
=
\sum_{v\in V_i}
\alpha_{iv}\widetilde h_{iv},
\qquad
\alpha_i
=
\mathrm{softmax}\left(s_i(\widetilde H_i)\right).
```

Both are invariant if the score map is equivariant and the weights are
reindexed with the rows.

## 4. When A Graph Collapses To A Set

If the adjacency is the complete graph and the context layer treats every
node identically, a graph layer can emulate a set-context operator. But this
does not mean every graph is a set in disguise. The input relation can carry
information that no permutation-invariant function of the feature multiset can
recover.

Take two nodes with scalar features:

```math
H
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
```

Let `A^(near)` connect the two nodes and let `A^(far)` contain only self-loops.
For the linear message map:

```math
\widetilde H
=
\widetilde A H,
```

with:

```math
\widetilde A^{(\mathrm{near})}
=
\begin{bmatrix}
0&1\\
1&0
\end{bmatrix},
\qquad
\widetilde A^{(\mathrm{far})}
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
```

the contextual states are:

```math
\widetilde H^{(\mathrm{near})}
=
\begin{bmatrix}
0\\
1
\end{bmatrix},
\qquad
\widetilde H^{(\mathrm{far})}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
```

The feature multiset is identical before context, but a downstream nodewise
readout can distinguish the two graphs. A set-only model cannot see the
difference because it never receives `A`.

This is the central graph-MIL inductive bias:

```text
the relation support is itself evidence, not merely a computational shortcut.
```

## 5. C/R/G/S Placement

```text
C:
    message passing, graph attention, typed context, or learned relation
    attention

R:
    sum, mean, max, global attention, typed pooling, or hierarchical transfer

G:
    coordinate graph, feature graph, learned directed graph, or hierarchy

S:
    slide classification, staging, subtype, survival, or localization loss
```

The same readout can have different semantics depending on `C`. Mean pooling
of raw patch embeddings preserves a patch first moment; mean pooling after
graph context preserves a first moment of neighborhood-conditioned features.
