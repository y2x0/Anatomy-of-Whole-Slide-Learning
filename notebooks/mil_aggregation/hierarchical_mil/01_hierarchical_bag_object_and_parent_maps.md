# Hierarchical Bag Object And Parent Maps

## 1. Flat bags versus nested bags

A flat slide bag is

`math
B_i=\{x_{ij}\}_{j=1}^{n_i},
\qquad
x_{ij}\in\mathbb R^{d_0}.
`

Its instance index has no intrinsic parent. A hierarchical bag introduces levels
of units:

`math
V_i^{(0)},V_i^{(1)},\ldots,V_i^{(L)},
\qquad
|V_i^{(\ell)}|=n_i^{(\ell)}.
`

Level zero contains the finest modeled units. For every
0\leq\ell<L, define a parent map

`math
\pi_i^{(\ell)}:V_i^{(\ell)}\longrightarrow V_i^{(\ell+1)}.
`

Each child has exactly one parent:

`math
\forall v\in V_i^{(\ell)},
\qquad
\pi_i^{(\ell)}(v)\in V_i^{(\ell+1)}.
`

This does not require every parent to have the same number of children. The
level structure may be a tree, a forest, or a hierarchy with an explicit
slide-level root.

## 2. Assignment matrices

After ordering the units, the hard parent map has an assignment matrix

`math
P_i^{(\ell)}
\in
\{0,1\}^{n_i^{(\ell)}\times n_i^{(\ell+1)}},
\qquad
P_{iab}^{(\ell)}
=
\mathbf 1\!\left[
\pi_i^{(\ell)}(v_{ia}^{(\ell)})
=
v_{ib}^{(\ell+1)}
\right].
`

The one-parent constraint is a row-stochastic condition:

`math
P_i^{(\ell)}\mathbf 1_{n_i^{(\ell+1)}}
=
\mathbf 1_{n_i^{(\ell)}}.
`

The number of children of parent b is

`math
m_{ib}^{(\ell)}
=
\sum_{a=1}^{n_i^{(\ell)}}P_{iab}^{(\ell)}.
`

A valid hard hierarchy has m_{ib}^{(\ell)}\geq 1 for every represented parent.
If empty parent slots are retained for batching, they require an explicit mask
rather than an implicit zero interpretation.

Soft assignment replaces P_i^{(\ell)} with nonnegative weights:

`math
A_i^{(\ell)}\in\mathbb R_{\geq 0}^{n_i^{(\ell)}\times n_i^{(\ell+1)}},
\qquad
A_i^{(\ell)}\mathbf 1=\mathbf 1.
`

A soft row can distribute one child across several parents. This changes the
object from a partition hierarchy to an overlapping cover or routing graph.

## 3. Hierarchical permutation symmetry

Let Q_i^{(\ell)} be a permutation matrix that reorders level-\ell units. A
reordering preserves the hierarchy only if the assignment matrix transforms as

`math
P_i^{(\ell)}
\longmapsto
Q_i^{(\ell)}
P_i^{(\ell)}
\left(Q_i^{(\ell+1)}\right)^{\mathsf T}.
`

A hierarchical encoder is equivariant when

`math
\Phi_{\theta}^{(\ell)}
\left(
Q_i^{(\ell)}H_i^{(\ell)},
Q_i^{(\ell)}P_i^{(\ell)}
\left(Q_i^{(\ell+1)}\right)^{\mathsf T}
\right)
=
Q_i^{(\ell+1)}
\Phi_{\theta}^{(\ell)}
\left(
H_i^{(\ell)},P_i^{(\ell)}
\right).
`

The allowable permutation group is therefore

`math
\mathfrak G(\mathcal H_i)
=
\left\{
(Q^{(0)},\ldots,Q^{(L)}):
Q^{(\ell)}P^{(\ell)}
=
P^{(\ell)}Q^{(\ell+1)}
\text{ for all }\ell
\right\}.
`

A flat set model admits independent permutations of all patches. A fixed
hierarchy admits only coupled permutations that preserve child-parent incidence.
This is the precise mathematical price of adding region identity.

## 4. Parent maps are part of the representation

Two slides can have the same fine-scale feature multiset but different parent
maps:

`math
\{x_{ij}\}_{j=1}^{n_i}
=
\{x_{i'j}\}_{j=1}^{n_{i'}},
\qquad
P_i^{(0)}\neq P_{i'}^{(0)}.
`

A hierarchical predictor can distinguish them because its output is a function of
(H,P), not H alone:

`math
\widehat y_i
=
F_{\theta}\left(H_i^{(0)},P_i^{(0)},\ldots,P_i^{(L-1)}\right).
`

If the maps are discarded after constructing coarse features, later layers
cannot recover distinctions in the fibers of the assignment operator. This is
why a claim that a model is hierarchical must specify whether hierarchy is used
only to construct tokens or remains available to the predictor.

## 5. C/R/G/S placement

`text
C: within-level attention, graph, or token context plus parent-conditioned
   cross-level composition.

R: coarsening at each parent map, followed by a final level-wise readout.

G: the parent maps, the level-wise supports, and any learned routing.

S: slide labels, region labels, pseudo-labels, or self-supervised cross-view
   targets; the hierarchy itself is not supervision unless assignments are
   learned from labels.
`

The defining surviving statistic is multiscale: a fine-scale statistic is
computed, compressed into parent units, and then transformed again before the
slide readout.
