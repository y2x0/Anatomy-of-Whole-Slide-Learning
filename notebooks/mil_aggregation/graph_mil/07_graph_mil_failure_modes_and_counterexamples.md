# Graph MIL Failure Modes And Counterexamples

Graph context adds an inductive bias before readout. Every added relation can
preserve a useful interaction or create a new collision. The right question is
not whether the graph is “interpretable,” but which perturbations leave the
task statistic unchanged.

## 1. Wrong Support Is A Representation Error

Let a one-layer linearized graph context be:

```math
H^{(1)}
=
\widetilde A H^{(0)}W,
```

where `A_tilde` contains whatever normalization and self-loops the model uses.
If the true task depends on a relation `A_star` but the model receives `A`,
then the first contextual statistic is:

```math
\widetilde A H^{(0)}
\neq
\widetilde A^{\star}H^{(0)}
```

in general, even when the patch encoder is perfect. More layers cannot guarantee
that a wrong support becomes correct; they only propagate the wrong relation.

This appears differently in the mapped papers:

```text
Patch-GCN:
    coordinate graph can miss long-range morphology or connect irrelevant
    nearby patches

HEAT:
    feature kNN can connect semantically similar but spatially unrelated tiles

WiKG:
    top-k learned support can be unstable or task-shortcut-driven

HACT:
    cell-to-tissue assignment can place fine evidence in the wrong coarse node
```

## 2. Same Features, Different Graphs

Use two nodes and one scalar feature:

```math
H^{(0)}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
```

Consider self-loop-normalized supports:

```math
\widetilde A_1
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
\widetilde A_2
=
\frac{1}{2}
\begin{bmatrix}
1&1\\
1&1
\end{bmatrix}.
```

Then:

```math
\widetilde A_1H^{(0)}
=
\begin{bmatrix}1\\0\end{bmatrix},
\qquad
\widetilde A_2H^{(0)}
=
\begin{bmatrix}1/2\\1/2\end{bmatrix}.
```

A graph model can distinguish the two slides even though their patch multiset
is identical. A set model cannot. Conversely, if the graph is an artifact, the
model can learn a label from an artifact that a set model would be unable to
see.

## 3. Oversmoothing

For a linear normalized propagation operator:

```math
H^{(\ell+1)}
=
\widetilde A H^{(\ell)}W_{\ell},
```

ignore feature transforms and write:

```math
H^{(L)}
=
\widetilde A^{L}H^{(0)}.
```

If `A_tilde` has dominant eigenvalue `1` and all other eigenvalues satisfy:

```math
|\lambda_r|<1,
```

then components orthogonal to the dominant eigenspace decay with depth:

```math
\widetilde A^{L}H^{(0)}
\longrightarrow
\Pi_{\mathrm{dom}}H^{(0)}.
```

Node states become difficult to distinguish. A downstream mean readout then
sees a field with less local identity, and attention scores can collapse toward
similar values.

Residual and dense connections, as used in Patch-GCN, preserve multiple scales
but do not make oversmoothing mathematically impossible.

## 4. Oversquashing

Suppose a node receives information from `B_L(v)` nodes within `L` hops but
has fixed hidden width `d`. The number of source degrees of freedom can grow
much faster than the dimension of the node state:

```math
\left|B_L(v)\right|d_{\mathrm{in}}
\gg
d.
```

The map from the distant neighborhood to the final node is then a compression:

```math
\mathbb{R}^{|B_L(v)|d_{\mathrm{in}}}
\longrightarrow
\mathbb{R}^{d}.
```

Distinct long-range configurations can produce the same node state. A graph
readout cannot recover interactions already lost inside the node bottleneck.
Sparse learned top-k support can reduce the number of routes, while dense
feature graphs can create short but semantically wrong routes.

## 5. Sum Readout And Size Shortcut

Suppose every contextualized node equals a common vector `u`:

```math
\widetilde h_{iv}=u
\qquad
\text{for all }v.
```

Then:

```math
z_i^{\mathrm{sum}}
=
n_i u,
\qquad
z_i^{\mathrm{mean}}
=
u.
```

The sum readout contains node count even when morphology is unchanged. HACT's
cell-to-tissue sum can intentionally preserve cellularity; a flat graph sum
may instead learn tissue segmentation density, patch count, or preprocessing
variation. Size should be treated as a modeled statistic, not an accidental
one.

## 6. Hard Top-K Instability

For WiKG, let the `K`th and `(K+1)`th scores for target `v` be:

```math
\gamma_{iv}
=
s_{iv,(K)}-s_{iv,(K+1)}.
```

When `gamma_iv` is small, a small parameter or feature perturbation can change
the support. Within a fixed support, softmax derivatives exist; at a rank tie,
the discrete graph map changes:

```math
s_{iv,(K)}=s_{iv,(K+1)}
\Longrightarrow
\text{possible edge replacement}.
```

This creates a topology-level sensitivity that is invisible if one reports only
the continuous attention weights after top-k selection.

## 7. Pseudo-Label And Hierarchy Errors

For HEAT, an upstream type error changes both the projection family and the PL
row:

```math
\tau_i(v)=a
\longrightarrow
\tau_i(v)=b.
```

The same patch can therefore affect:

```text
the HEAT Q/K/V maps;
the edge-conditioned message;
the type-wise pooled row.
```

For HACT, assignment pooling has an explicit nullspace. If:

```math
B_i^{\top}\Delta=0,
```

then:

```math
B_i^{\top}(H+\Delta)=B_i^{\top}H.
```

Fine-scale differences in that nullspace are unobservable to the tissue graph.

## 8. Deletion Must Match The Operator

Deleting a node from a graph changes more than one input feature. It can change
incident messages, degree normalization, top-k ranks, typed group counts, and
the graph readout. Define a graph deletion intervention:

```math
\Delta_{iv}^{\mathrm{graph}}
=
F_{\theta}(G_i)
-
F_{\theta}(G_i\setminus v).
```

This is not equivalent to setting `h_iv` to zero while holding adjacency fixed:

```math
F_{\theta}(G_i\setminus v)
\neq
F_{\theta}(H_i^{(0)}\text{ with row }v\text{ zeroed},A_i)
```

in general. The intervention should be named precisely when used as an
explanation or sanity check.
