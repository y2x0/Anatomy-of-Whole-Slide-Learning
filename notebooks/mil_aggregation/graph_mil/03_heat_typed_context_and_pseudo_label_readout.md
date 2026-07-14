# HEAT: Heterogeneous Edge-Attribute Context and Pseudo-Label Pooling

Source:

Chan et al., Histopathology Whole Slide Image Analysis with Heterogeneous Graph
Representation Learning, CVPR 2023.

[arXiv](https://arxiv.org/abs/2307.04189)

[CVPR open-access paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chan_Histopathology_Whole_Slide_Image_Analysis_With_Heterogeneous_Graph_Representation_Learning_CVPR_2023_paper.pdf)

The paper's WSI object is a heterogeneous graph with:

1. a node feature for each selected patch;
2. a nucleus-type attribute assigned by a pretrained HoverNet teacher;
3. an edge support obtained by feature-space k-nearest neighbors;
4. a continuous edge attribute given by Pearson correlation of endpoint features;
5. a heterogeneous edge-attribute transformer called HEAT;
6. pseudo-label pooling that groups nodes by a fixed semantic type vocabulary.

The forward map is:

```math
\text{WSI}
\longrightarrow
\text{typed attributed graph}
\longrightarrow
\text{HEAT node context}
\longrightarrow
\text{type-indexed PL pooling}
\longrightarrow
\text{graph-level task head}.
```

The key representation claim is:

```math
\boxed{
\text{the graph is heterogeneous in both nodes and edges, and the pooling index
has a fixed semantic meaning}
}
```

This is different from a homogeneous graph followed by ordinary mean or
attention pooling. It is also different from learned cluster pooling, where
cluster identities can change meaning across slides.

## 1. Heterogeneous graph definition

Let a WSI graph be:

```math
G
=
(V,E,\mathcal A,\mathcal R,X,\tau,\phi).
```

Here:

| Symbol | Meaning |
| --- | --- |
| V | patch-node set |
| E | directed or stored edge set |
| A | node-type vocabulary |
| R | edge-attribute space |
| X | node-feature space |
| tau | node-type map |
| phi | edge-attribute map |

For every node v:

```math
\tau(v)
\in
\mathcal A.
```

For every edge e:

```math
\phi(e)
\in
\mathcal R.
```

The node feature is:

```math
x_v
\in
\mathbb R^{d_0}.
```

An edge can be written as:

```math
e=(s,t),
```

where s is the source node and t is the target node. The source-target
orientation matters when messages are normalized over incoming neighbors.

The heterogeneous graph is not defined only by a feature matrix and adjacency.
It includes the maps tau and phi:

```math
G
\ne
(X,A)
\quad
\text{unless node and edge attributes are omitted by design}.
```

## 2. Patch construction

The paper first applies Otsu thresholding and a sliding-window procedure to the
WSI. Background patches are removed. Let the retained patch images be:

```math
\{x_v\}_{v\in V},
\qquad
x_v
\in
\mathbb R^{H\times W\times3}.
```

The source-level pipeline is:

```text
downsampled WSI
    -> Otsu tissue/background threshold
    -> non-overlapping patch extraction
    -> removal of uninformative background patches
    -> graph nodes
```

The exact patch size and magnification belong to the implementation and dataset
configuration. The mathematical object needed by HEAT is the retained node set
and its feature/type/edge attributes.

A pretrained KimiaNet encoder maps each retained patch to a node feature:

```math
h_v^{(0)}
=
f_{\mathrm{KimiaNet}}(x_v)
\in
\mathbb R^{d_0}.
```

Stack the features:

```math
H^{(0)}
=
\begin{bmatrix}
h_1^{(0)\mathsf T}\\
\vdots\\
h_{|V|}^{(0)\mathsf T}
\end{bmatrix}
\in
\mathbb R^{|V|\times d_0}.
```

KimiaNet supplies the node representation. It does not supply the semantic
nucleus type used by PL pooling.

## 3. HoverNet-derived pseudo-types

The paper uses pretrained HoverNet with PanNuke knowledge to detect and classify
nuclei inside patches. Let the nuclei detected in patch v be:

```math
\mathcal N_v
=
\{n_{v,r}\}_{r=1}^{N_v}.
```

Each detected nucleus receives a teacher type:

```math
\widehat t_{v,r}
\in
\mathcal A.
```

The patch node type is assigned by majority vote:

```math
\tau(v)
=
\arg\max_{a\in\mathcal A}
\sum_{r=1}^{N_v}
\mathbf 1
\{\widehat t_{v,r}=a\}.
```

If no nucleus is detected or the vote is tied, the implementation needs a
specified fallback. The paper's figure includes an unlabeled or no-label
category; a faithful abstraction is to include a background/unknown type in
the vocabulary when the pipeline uses one:

```math
\mathcal A
=
\mathcal A_{\mathrm{nuclei}}
\cup
\{\mathrm{unknown}\}.
```

The type is a teacher-generated attribute:

```math
\tau(v)
=
\Psi_{\mathrm{HoverNet}}
\left(
x_v
\right).
```

It is not an observed pathologist annotation for the slide-level task. A
semantic row in PL pooling can be stable while still being wrong for a
particular patch.

## 4. Feature-space kNN support

HEAT constructs edges from node-feature similarity rather than physical
coordinate proximity. Let d be a feature distance. The source uses k-nearest
neighbors:

```math
\mathcal N(v)
=
\mathrm{KNN}_k
\left(
h_v^{(0)};
\{h_u^{(0)}\}_{u\ne v}
\right).
```

The support is:

```math
E
=
\{(u,v):u\in\mathcal N(v)\}.
```

The direction convention can be stated as source u pointing to target v. The
important geometry is:

```math
u\in\mathcal N(v)
\quad
\Longleftrightarrow
\quad
h_u^{(0)}
\text{ is among the most similar features to }h_v^{(0)}.
```

This differs from a coordinate graph:

```math
\mathcal N_{\mathrm{coord}}(v)
=
\mathrm{KNN}_k(c_v;C).
```

HEAT's graph can connect spatially distant patches with similar KimiaNet
features. Physical adjacency is not the relation used by the source graph
construction unless coordinates are added in a separate implementation.

## 5. Pearson edge attribute

For edge e equal to u-to-v, compute a scalar semantic similarity attribute:

```math
r_{uv}
=
\rho_{\mathrm{Pearson}}
\left(
h_u^{(0)},
h_v^{(0)}
\right).
```

For vectors h-u and h-v with coordinate means h-bar-u and h-bar-v:

```math
\rho_{\mathrm{Pearson}}
\left(
h_u^{(0)},
h_v^{(0)}
\right)
=
\frac{
\sum_{r=1}^{d_0}
\left(
h_{u,r}^{(0)}-\overline h_u^{(0)}
\right)
\left(
h_{v,r}^{(0)}-\overline h_v^{(0)}
\right)
}{
\sqrt{
\sum_{r=1}^{d_0}
\left(
h_{u,r}^{(0)}-\overline h_u^{(0)}
\right)^2
}
\sqrt{
\sum_{r=1}^{d_0}
\left(
h_{v,r}^{(0)}-\overline h_v^{(0)}
\right)^2
}
}.
```

The edge attribute is a continuous relation:

```math
\phi(u,v)
=
r_{uv}
\in
\mathbb R.
```

The heterogeneous graph therefore has two kinds of heterogeneity:

```math
\text{node heterogeneity}
=
\tau(v),
```

```math
\text{edge heterogeneity}
=
\phi(u,v).
```

The edge support came from feature kNN, while the Pearson value is an additional
relation attribute. Similarity used to choose an edge and similarity stored on
the edge are related but not identical operations.

## 6. Edge and node augmentation

The paper uses data augmentation such as random edge removal and node removal
to reduce sensitivity to noisy graph construction. Let a random mask be:

```math
m_{uv}^{E}
\in
\{0,1\},
\qquad
m_v^{V}
\in
\{0,1\}.
```

The augmented edge support is:

```math
E'
=
\{(u,v)\in E:m_{uv}^{E}=1,\;m_u^{V}=1,\;m_v^{V}=1\}.
```

This changes the context graph seen during training:

```math
G'
=
(V',E',\mathcal A,\mathcal R,X',\tau',\phi').
```

The augmentation is not a new biological relation. It is a regularizer for
uncertain support and teacher-generated attributes.

A model robust to edge pruning may rely on distributed relational evidence. A
model that collapses after one edge deletion may be using a brittle topological
shortcut.

## 7. Input to a HEAT layer

At layer ell minus one, each node has:

```math
H_v^{(\ell-1)}
\in
\mathbb R^{d_{\ell-1}}.
```

Each edge has an attribute or latent edge state:

```math
h_{e}^{(\ell-1)}
\in
\mathbb R^{d_e}.
```

The node type is fixed by the teacher vocabulary:

```math
\tau(v)
\in
\mathcal A.
```

For each attention head r, HEAT uses node-type-specific projection matrices.
The source notation is easiest to state for an edge from source s to target t.

## 8. Type-specific key, query, and value

For head r, source-key projection:

```math
k_{e}^{(r)}
=
W_{\tau(s)}^{(r)}
H_s^{(\ell-1)}.
```

Source-value projection:

```math
v_{e}^{(r)}
=
W_{\tau(s)}^{(r)}
H_s^{(\ell-1)}.
```

Target-query projection:

```math
q_{e}^{(r)}
=
W_{\tau(t)}^{(r)}
H_t^{(\ell-1)}.
```

The type-specific matrices map heterogeneous node feature distributions into a
common head space:

```math
W_a^{(r)}
:
\mathbb R^{d_{\ell-1}}
\longrightarrow
\mathbb R^{d_h},
\qquad
a\in\mathcal A.
```

The source and target can use different type-specific projections:

```math
\tau(s)\ne\tau(t)
\Longrightarrow
W_{\tau(s)}^{(r)}
\ne
W_{\tau(t)}^{(r)}
\quad
\text{in general}.
```

This is the node heterogeneity in HEAT. A homogeneous GAT would typically
share a single projection family across node types.

## 9. Edge attribute projection

Project the edge state:

```math
e_{st}^{\prime}
=
W_{\mathrm{edge}}
h_{st}^{(\ell-1)}.
```

For a scalar initial Pearson attribute, W-edge can lift the scalar to the head
dimension:

```math
W_{\mathrm{edge}}
:
\mathbb R
\longrightarrow
\mathbb R^{d_h}.
```

For a high-dimensional edge state, the same equation maps its current dimension
to d-h.

The edge feature is not merely concatenated after message aggregation. It enters
the attention score:

```math
\mathrm{edge}
\longrightarrow
\text{compatibility score}
\longrightarrow
\text{message weight}.
```

## 10. HEAT compatibility score

The paper's head-wise attention score is:

```math
\mathrm{ATT}(e,r)
=
\frac{
\left(
k_e^{(r)}
\odot
e_{st}^{\prime}
\right)^{\mathsf T}
q_e^{(r)}
}{
\sqrt{d_h}
}.
```

Equivalently:

```math
\mathrm{ATT}(e,r)
=
\frac{
\left\langle
k_e^{(r)}
\odot
e_{st}^{\prime},
q_e^{(r)}
\right\rangle
}{
\sqrt{d_h}
}.
```

The elementwise product is the edge-conditioned modulation of the key-query
compatibility. If e-prime is all ones, the score reduces to ordinary scaled
dot-product attention:

```math
e_{st}^{\prime}
=
\mathbf 1
\Longrightarrow
\mathrm{ATT}(e,r)
=
\frac{
\left(k_e^{(r)}\right)^{\mathsf T}
q_e^{(r)}
}{
\sqrt{d_h}
}.
```

For a scalar edge attribute lifted through W-edge, different edge values
change the direction and magnitude of the modulated key.

The score depends on:

```math
\mathrm{ATT}(e,r)
=
\mathrm{ATT}
\left(
H_s^{(\ell-1)},
H_t^{(\ell-1)},
h_{st}^{(\ell-1)},
\tau(s),
\tau(t)
\right).
```

This is why HEAT is not merely a GAT with a type label appended to node
features. Node types choose projection families and edge state changes the
compatibility itself.

## 11. Multi-head edge attention

Concatenate the head-wise scores:

```math
\mathrm{ATT}_{\mathrm{all}}(e)
=
\mathop{\mathrm{Concat}}_{r=1}^{R}
\mathrm{ATT}(e,r).
```

For target t, normalize over all incoming source nodes:

```math
\alpha_{st}
=
\mathrm{softmax}_{s\in\mathcal N(t)}
\left(
\mathrm{ATT}_{\mathrm{all}}(s,t)
\right).
```

The normalization is over incoming edges, not over all edges in the graph:

```math
\sum_{s\in\mathcal N(t)}
\alpha_{st}
=
1.
```

If the concatenated attention score is vector-valued, the implementation must
specify whether softmax is taken over sources separately for each head or after
head concatenation. The paper's algorithm presents concatenated head scores
followed by a softmax over incoming edges. The invariant source-level claim is
that incoming edge weights are normalized per target and use all attention
heads.

## 12. Node update

The source value vectors are multiplied by the edge attention and aggregated:

```math
H_t^{(\ell)}
=
\bigoplus_{s\in\mathcal N(t)}
\left(
\mathop{\mathrm{Concat}}_{r=1}^{R}
\left(
v_{st}^{(r)}
\right)
\odot
\alpha_{st}
\right).
```

Here the aggregation operator can be mean aggregation, as described in the
paper. If the multi-head value is concatenated into one vector, the update has
dimension:

```math
H_t^{(\ell)}
\in
\mathbb R^{R d_h}.
```

An optional output projection can map this to the next layer width:

```math
H_t^{(\ell)}
\longmapsto
H_t^{(\ell)}W_O.
```

The source algorithm returns updated node features and latent edge features.

## 13. Edge-state update

The transformed edge state is retained as a latent edge feature:

```math
h_{st}^{(\ell)}
=
e_{st}^{\prime}
=
W_{\mathrm{edge}}h_{st}^{(\ell-1)}.
```

The exact implementation can include additional edge updates, but the paper's
algorithm explicitly computes the projected edge feature and returns it as the
updated edge state.

The edge pathway is therefore recurrent across HEAT layers:

```math
h_{st}^{(0)}
\longrightarrow
h_{st}^{(1)}
\longrightarrow
\cdots
\longrightarrow
h_{st}^{(L)}.
```

A fixed Pearson scalar is not the only edge object at later layers. The layer
learns a latent edge representation from the initial semantic similarity.

## 14. HEAT as a typed edge-conditioned message map

The complete layer can be summarized:

```math
\left(
H^{(\ell-1)},h_E^{(\ell-1)},\tau
\right)
\longrightarrow
\left(
K,Q,V,E'
\right)
\longrightarrow
\mathrm{ATT}_{\mathrm{all}}
\longrightarrow
\alpha
\longrightarrow
H^{(\ell)},h_E^{(\ell)}.
```

The message from s to t is:

```math
M_{st}
=
\alpha_{st}
\mathop{\mathrm{Concat}}_{r=1}^{R}
v_{st}^{(r)}.
```

The target state is a neighbor aggregation:

```math
H_t^{(\ell)}
=
\bigoplus_{s\in\mathcal N(t)}M_{st}.
```

The message depends on node types through K/Q/V and on edge attributes through
the compatibility score. The graph context operator is:

```math
\mathcal C_{\mathrm{HEAT}}
=
\left(
H^{(\ell)},h_E^{(\ell)}
\right)_{\ell=1}^{L}.
```

## 15. Permutation equivariance

Let P be a node permutation. Transform:

```math
H'
=
PH,
\qquad
A'
=
PAP^{\mathsf T},
\qquad
\tau'
=
P\tau.
```

Edge attributes must be reindexed with their endpoints:

```math
h_{P(s)P(t)}^{\prime}
=
h_{st}.
```

Because type-specific projections are selected by the reindexed type map, the
source-key, target-query, and source-value vectors are reindexed rather than
changed in semantic identity:

```math
K'
=
PK,
\qquad
Q'
=
PQ,
\qquad
V'
=
PV.
```

Incoming-neighbor softmax is invariant to the order of the incoming set:

```math
\alpha_{P(s)P(t)}'
=
\alpha_{st}.
```

The node update is equivariant:

```math
H^{(\ell)\prime}
=
PH^{(\ell)}.
```

The PL pooling rows are invariant to node order because nodes are grouped by
their type value, not by storage position.

## 16. Pseudo-label pooling

After L HEAT layers, let:

```math
\widetilde H
=
\{\widetilde h_v\}_{v\in V}.
```

For type a in the fixed vocabulary A:

```math
V_a
=
\{v\in V:\tau(v)=a\}.
```

Collect the type-specific feature matrix:

```math
X_a
=
\begin{bmatrix}
\widetilde h_v^{\mathsf T}
\end{bmatrix}_{v\in V_a}
\in
\mathbb R^{|V_a|\times d_L}.
```

The per-type readout is a learned readout layer:

```math
h_a
=
\mathcal R_a(X_a).
```

The paper's PL-Pool algorithm initializes one readout layer for every node type,
pools the feature matrix for that type, and assigns the result to row a of a
matrix S.

The output is:

```math
S
=
\begin{bmatrix}
h_{a_1}^{\mathsf T}\\
\vdots\\
h_{a_{|\mathcal A|}}^{\mathsf T}
\end{bmatrix}
\in
\mathbb R^{|\mathcal A|\times d_{\mathrm{PL}}}.
```

The row coordinate is semantic:

```math
S_{a,:}
=
h_a
\quad
\text{means the pooled representation of teacher type }a.
```

This fixed coordinate system is the central difference from learned cluster
pooling.

## 17. Per-type readout choices

The source describes a readout layer for each type and a later graph-level
readout. A simple mean realization is:

```math
h_a^{\mathrm{mean}}
=
\frac{1}{|V_a|}
\sum_{v\in V_a}
\widetilde h_v,
\qquad
|V_a|>0.
```

A sum realization is:

```math
h_a^{\mathrm{sum}}
=
\sum_{v\in V_a}
\widetilde h_v.
```

An attention realization is:

```math
h_a^{\mathrm{attn}}
=
\sum_{v\in V_a}
\alpha_{v|a}\widetilde h_v,
```

with:

```math
\alpha_{v|a}
=
\frac{
\exp(s_{v|a})
}{
\sum_{u\in V_a}\exp(s_{u|a})
}.
```

The choice changes the within-type surviving statistic. PL pooling fixes the
semantic grouping; it does not uniquely determine the per-type statistic.

## 18. Empty-type masks

A slide may contain no node assigned to type a. Define:

```math
m_a
=
\mathbf 1\{|V_a|>0\}.
```

A masked type representation can be:

```math
\widetilde h_a
=
m_a h_a.
```

The graph-level readout can receive both S and m:

```math
z
=
\mathcal R_{\mathrm{graph}}(S,m).
```

For a mean over type rows, an unmasked mean is:

```math
z_{\mathrm{unmasked}}
=
\frac{1}{|\mathcal A|}
\sum_{a\in\mathcal A}h_a.
```

A masked mean is:

```math
z_{\mathrm{masked}}
=
\frac{
\sum_{a\in\mathcal A}m_a h_a
}{
\sum_{a\in\mathcal A}m_a
},
```

provided at least one type is present.

These are different representations. An all-zero row can mean “absent type,”
“zero pooled feature,” or “padding,” so the mask should be explicit if the
implementation distinguishes those cases.

## 19. Graph-level readout

The PL matrix S is not necessarily the final graph feature. Apply a second
readout:

```math
z
=
\mathcal R_{\mathrm{graph}}(S).
```

For mean readout over rows:

```math
z^{\mathrm{mean}}
=
\frac{1}{|\mathcal A|}
\sum_{a\in\mathcal A}h_a.
```

For a learned row attention:

```math
\gamma_a
=
\frac{
\exp(g(h_a))
}{
\sum_{b\in\mathcal A}\exp(g(h_b))
},
```

```math
z^{\mathrm{attn}}
=
\sum_{a\in\mathcal A}
\gamma_a h_a.
```

The full PL pooling composition is:

```math
\widetilde H
\longrightarrow
\{X_a\}_{a\in\mathcal A}
\longrightarrow
\{h_a\}_{a\in\mathcal A}
\longrightarrow
S
\longrightarrow
z.
```

The graph-level statistic retains type-stratified summaries and whatever
information the row readout preserves.

## 20. Why fixed semantic rows reduce over-parameterization

Learned cluster pooling often computes assignments:

```math
Q
=
\mathrm{softmax}
\left(
\widetilde H W_Q
\right)
\in
\mathbb R^{|V|\times K}.
```

Cluster k has no fixed semantic identity across graphs:

```math
Q_{:,k}
\quad
\text{can represent different morphology on different slides}.
```

PL pooling instead defines:

```math
\text{cluster }a
=
\{v:\tau(v)=a\},
```

where a has the same teacher-defined meaning across slides.

The parameter and identifiability problem changes from learning arbitrary
cluster assignment semantics to learning a readout for a fixed semantic
partition:

```math
\mathcal R_a
:
\mathbb R^{|V_a|\times d_L}
\longrightarrow
\mathbb R^{d_{\mathrm{PL}}}.
```

The fixed row coordinate does not make the teacher type correct. It makes the
cluster identity comparable.

## 21. What PL pooling preserves

The output S preserves:

```text
one learned summary for each teacher-defined node type
the fixed row identity of that type
the graph-level readout of those summaries
```

It generally discards:

```text
the order of nodes within a type
the spatial arrangement within a type
the full type-conditioned distribution unless the per-type readout preserves it
teacher uncertainty if only the argmax type is retained
the identity of individual nodes after pooling
```

Two graphs can collide:

```math
S(G)
=
S(G')
\quad
\text{while}
\quad
G\ne G'.
```

The collision class depends on the per-type readout and the final graph readout.

## 22. Type-count information

If the per-type readout is a normalized mean:

```math
h_a
=
\frac{1}{|V_a|}
\sum_{v\in V_a}\widetilde h_v,
```

then the type count is not retained except through a mask or separate feature.

If the per-type readout is a sum:

```math
h_a
=
\sum_{v\in V_a}\widetilde h_v,
```

then type count and feature magnitude are coupled.

An explicit type-count vector is:

```math
c
=
\left(
|V_{a_1}|,\ldots,|V_{a_{|\mathcal A|}}|
\right).
```

A count-aware graph representation can be:

```math
z
=
\mathcal R_{\mathrm{graph}}(S,c,m).
```

The paper's high-level PL-Pool claim is semantic-consistent grouping. It does
not imply that every count statistic is preserved by every readout choice.

## 23. Teacher error and semantic contamination

Let the latent biological type be Z-v and teacher type be tau-v. The teacher
confusion event is:

```math
\tau(v)\ne Z_v.
```

For a type a, the observed group is:

```math
V_a
=
\{v:\tau(v)=a\}.
```

The group can contain latent types b different from a:

```math
V_a
\cap
\{v:Z_v=b\}
\ne
\varnothing.
```

The pooled row is then a mixture:

```math
h_a
=
\mathcal R_a
\left(
\{\widetilde h_v:v\in V_a\}
\right).
```

PL pooling makes the mixture consistently indexed, but does not eliminate the
mixture. A fixed row can be a consistently wrong semantic statistic.

## 24. Type imbalance

Let n-a equal the number of nodes in type a:

```math
n_a
=
|V_a|.
```

A rare type has:

```math
n_a
\ll
|V|.
```

A mean readout over the rare type has high estimator variance. If node features
within type a have covariance Sigma-a, the sample mean covariance is
approximately:

```math
\mathrm{Cov}
\left(
h_a^{\mathrm{mean}}
\right)
=
\frac{\Sigma_a}{n_a},
```

under independent sampling as a diagnostic approximation.

The graph nodes are not necessarily independent, so spatial or graph
correlation can make the effective count smaller. Define an effective count
n-a-eff when within-type observations are correlated:

```math
n_{a,\mathrm{eff}}
=
\frac{n_a}{
1+
2\sum_{r\ge1}\rho_{a,r}
},
```

as a conceptual time-series-style diagnostic along a chosen graph ordering.
The exact estimator depends on the graph sampling scheme. The important point
is that rare types and correlated nodes make type rows statistically unstable.

## 25. HEAT versus ordinary GAT

A homogeneous graph attention score can be written:

```math
\mathrm{ATT}_{\mathrm{GAT}}(s,t)
=
\frac{
\left(W h_s\right)^{\mathsf T}
\left(W h_t\right)
}{
\sqrt{d_h}
}.
```

HEAT uses type-specific projections and edge modulation:

```math
\mathrm{ATT}_{\mathrm{HEAT}}(s,t,r)
=
\frac{
\left(
W_{\tau(s)}^{(r)}h_s
\odot
W_{\mathrm{edge}}h_{st}
\right)^{\mathsf T}
W_{\tau(t)}^{(r)}h_t
}{
\sqrt{d_h}
}.
```

The difference is structural:

```text
GAT:
    shared node projection, no explicit edge attribute in the score

HEAT:
    source/target type-specific projections, edge-conditioned score
```

If all node types share W and all edge states are the all-ones vector, HEAT
reduces toward ordinary scaled dot-product graph attention:

```math
W_a=W
\quad
\forall a,
\qquad
W_{\mathrm{edge}}h_{st}
=
\mathbf 1
\Longrightarrow
\mathrm{ATT}_{\mathrm{HEAT}}
=
\mathrm{ATT}_{\mathrm{GAT}}.
```

This is a limiting case, not the general architecture.

## 26. Feature graph versus coordinate graph

HEAT support:

```math
E_{\mathrm{HEAT}}
=
\mathrm{KNN}(H^{(0)}).
```

Patch-GCN support:

```math
E_{\mathrm{PatchGCN}}
=
\mathrm{KNN}(C).
```

The representations can differ even with identical node features:

```math
\mathcal C_{\mathrm{HEAT}}
\left(
H,E_{\mathrm{HEAT}},\tau,\phi
\right)
\ne
\mathcal C_{\mathrm{PatchGCN}}
\left(
H,E_{\mathrm{PatchGCN}}
\right).
```

HEAT's feature graph may capture recurring morphology across distant regions.
It may also capture encoder or stain shortcuts. A coordinate heatmap should
not be inferred from HEAT unless coordinates are explicitly included.

## 27. Relation to set MIL

A flat set model can compute:

```math
z_{\mathrm{set}}
=
\mathcal R
\left(
\{h_v\}_{v\in V}
\right).
```

HEAT first computes:

```math
\widetilde h_v
=
\mathcal C_{\mathrm{HEAT}}
\left(
h_v,
\{h_u,h_{uv},\tau(u),\tau(v):u\in\mathcal N(v)\}
\right),
```

then applies PL pooling:

```math
z_{\mathrm{HEAT}}
=
\mathcal R_{\mathrm{graph}}
\left(
\left\{
\mathcal R_a
\left(
\{\widetilde h_v:\tau(v)=a\}
\right)
\right\}_{a\in\mathcal A}
\right).
```

A set model can match the output only if it can infer the same relational and
type-conditioned statistic from the feature multiset alone. If the edge
attribute or type map contains information not recoverable from H, the set
model cannot reproduce it.

## 28. Causal-driven localization boundary

The paper motivates a Granger-causality-based localization procedure. A generic
information comparison is:

```math
\mathcal I
=
\text{all available graph information},
```

```math
\mathcal I_{-v}
=
\mathcal I
\setminus
\{\text{information associated with node }v\}.
```

A node is considered predictive under a Granger-style criterion if the model
predicts better with the information than without it:

```math
\mathrm{Perf}
\left(
\mathcal I
\right)
>
\mathrm{Perf}
\left(
\mathcal I_{-v}
\right).
```

This is a predictive intervention criterion. It is not automatically biological
causality. Removing a node can also change graph connectivity, type counts,
edge attention, and PL rows.

A graph-aware deletion score is:

```math
\Delta_v^{(q)}
=
q(G)
-
q(G_{-v}),
```

where G-minus-v specifies how incident edges, type rows, and normalization are
recomputed.

A PL row attribution can be different:

```math
\Delta_a^{(q)}
=
q(S)
-
q(S_{-a}).
```

Deleting a type row is not the same as deleting all nodes of that type from the
graph and recomputing HEAT.

## 29. Task head

For classification:

```math
\widehat y
=
\mathrm{softmax}
\left(
W_{\mathrm{cls}}z+b_{\mathrm{cls}}
\right).
```

For a binary task:

```math
\widehat y
=
\sigma
\left(
w_{\mathrm{cls}}^{\mathsf T}z+b_{\mathrm{cls}}
\right).
```

The graph representation and type rows can support classification, staging, or
other slide-level tasks. The target-specific score is:

```math
q
\in
\{
\ell_{\mathrm{class}},
\Pr(Y=1),
\ell_{\mathrm{stage},c}
\}.
```

A node type row with high attention in a downstream head is not a calibrated
probability that the row is causally responsible. The same explanation
boundary from ordinary attention MIL remains.

## 30. Prior-knowledge regularization

The paper motivates node types and edge attributes as prior knowledge useful
under scarce and high-dimensional WSI data. Abstractly, the graph prior is:

```math
\Pi
=
\left(
\tau,\phi
\right).
```

The learned predictor is:

```math
f_\theta
\left(
H,E;\Pi
\right).
```

The prior can reduce variance if it is informative:

```math
\mathrm{Var}
\left(
f_\theta\mid\Pi
\right)
<
\mathrm{Var}
\left(
f_\theta
\right)
\quad
\text{under a valid prior}.
```

It can increase bias if the teacher or edge similarity is wrong:

```math
\mathrm{Bias}
\left(
f_\theta\mid\Pi
\right)
>
\mathrm{Bias}
\left(
f_\theta
\right).
```

This is the regularization tradeoff. A semantic type label is not free
information; it is a noisy prior supplied by another model.

## 31. C/R/G/S placement

| Component | HEAT instantiation |
| --- | --- |
| C | type-specific query/key/value projections, edge-state projection, edge-conditioned scaled dot-product attention, incoming-edge aggregation |
| R | per-type PL readouts followed by a graph-level readout over the fixed type matrix |
| G | feature-space kNN support, node pseudo-types, Pearson edge attributes, and optional edge/node pruning during training |
| S | slide classification or staging labels; HoverNet/PanNuke types are generated upstream teacher attributes |

The complete map is:

```math
H^{(0)}
\xrightarrow{
\Gamma_{\mathrm{feature\ kNN},\mathrm{HoverNet},\mathrm{Pearson}}
G
}
\xrightarrow{
\mathcal C_{\mathrm{HEAT}}
}
\widetilde H
\xrightarrow{
\mathcal R_{\mathrm{PL}}
}
S
\xrightarrow{
\mathcal R_{\mathrm{graph}}
}
z
\xrightarrow{
\mathcal H
}
\widehat y.
```

The type map affects both C and R. The edge attribute affects C. The fixed row
coordinate is part of R.

## 32. Sanity checks

### 32.1 Type permutation test

Permute storage order while transforming the graph and type map:

```math
H'
=
PH,
\qquad
A'
=
PAP^{\mathsf T},
\qquad
\tau'
=
P\tau.
```

The graph output should be unchanged:

```math
z(G')
=
z(G).
```

### 32.2 Type semantic swap test

Swap two type labels without changing node features:

```math
\tau'(v)
=
\pi_{\mathrm{type}}(\tau(v)).
```

The output should generally change because type-specific projections and PL row
identities have changed. This tests whether node heterogeneity is actually used.

### 32.3 Edge attribute ablation

Set all edge states to a common value:

```math
h_{st}^{(0)}
=
h_0
\quad
\forall(s,t)\in E.
```

Compare with the original edge attributes. If the output is unchanged, the
edge-conditioned score may not be active.

### 32.4 Feature graph versus coordinate graph

Hold H fixed and replace the support:

```math
E_{\mathrm{feat}}
\longrightarrow
E_{\mathrm{coord}}.
```

A HEAT implementation should show a representation change if its graph context
uses support.

### 32.5 Teacher shuffle

Randomly permute node types within a slide while preserving counts. This keeps
the type histogram but destroys node-type alignment with features. A real
type-conditioned model should change:

```math
z(\tau)
\ne
z(\tau_{\mathrm{shuffle}})
```

in general.

### 32.6 Empty type

Remove all nodes of a type and verify the mask convention. The representation
must not silently confuse absence with a real zero-valued type row.

### 32.7 Edge-pruning stability

Sample two training-time edge masks and compare:

```math
z(G\odot M_E)
\quad
\text{and}
\quad
z(G\odot M_E').
```

Large variation under small pruning rates indicates dependence on brittle
single-edge relations.

### 32.8 PL collision

Construct two graphs with the same per-type pooled rows but different
within-type arrangements. If HEAT has no downstream access to those arrangements,
the graph-level outputs must collide:

```math
S(G)
=
S(G')
\Longrightarrow
z(G)
=
z(G').
```

## 33. Bottom line

HEAT is a typed, edge-conditioned graph context operator followed by a
semantically indexed pooling operator:

```math
\boxed{
\mathrm{HEAT\!-\!PL}
=
\mathcal R_{\mathrm{graph}}
\circ
\mathcal R_{\mathrm{type}}
\circ
\mathcal C_{\mathrm{edge\ attribute,\ type}}
}
```

The paper-specific objects are:

```text
KimiaNet patch features
feature-space kNN support
Pearson endpoint-feature edge attributes
HoverNet/PanNuke-derived majority-vote node types
type-specific key/query/value projections
edge-conditioned attention score
fixed-row pseudo-label pooling
graph-level readout over the type matrix
```

The surviving statistic is:

```math
S
=
\left[
\mathcal R_{a_1}(\widetilde H_{a_1});
\ldots;
\mathcal R_{a_{|\mathcal A|}}(\widetilde H_{a_{|\mathcal A|}})
\right].
```

The strongest inductive bias is semantic consistency: row a means the same
teacher-defined type across slides. The main failure modes are teacher
misclassification, feature-graph shortcuts, edge-attribute noise, rare-type
variance, missing-type conventions, and the loss of within-type arrangement.

The C/R/G/S summary is:

```math
\boxed{
\text{typed and edge-conditioned context}
+
\text{fixed semantic row pooling}
+
\text{feature-similarity geometry}
+
\text{slide-level supervision with teacher-generated attributes}
}
```

The right question is not whether HEAT has “better attention.” It is:

```math
\text{which typed edge relations and type-conditioned summaries survive the
HEAT-to-PL bottleneck?}
```
