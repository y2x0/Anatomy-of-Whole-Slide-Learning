# Graph MIL as a Graph Object: Equivariance, Invariance, and the Set Boundary

Sources:

- Gilmer et al., Neural Message Passing for Quantum Chemistry.
  [arXiv](https://arxiv.org/abs/1704.01212)
- Kipf and Welling, Semi-Supervised Classification with Graph Convolutional
  Networks. [arXiv](https://arxiv.org/abs/1609.02907)
- Veličković et al., Graph Attention Networks.
  [arXiv](https://arxiv.org/abs/1710.10903)
- Chen et al., Whole Slide Images are 2D Point Clouds: Context-Aware Survival
  Prediction using Patch-based Graph Convolutional Networks.
  [arXiv](https://arxiv.org/abs/2107.13048)

This note defines the mathematical object consumed by graph MIL. The main
question is not whether a model contains a graph layer. It is:

```math
\text{what relation between patches is made available to context propagation?}
```

A graph MIL model has four separable parts:

```math
\text{patch features}
\longrightarrow
\text{graph object}
\longrightarrow
\text{equivariant node context}
\longrightarrow
\text{invariant slide readout}.
```

The graph relation can encode physical proximity, feature similarity, learned
compatibility, edge type, or a fine-to-coarse assignment. These relations are
different inductive biases and should not be collapsed into one generic
“spatial graph” category.

## 1. From a WSI bag to a graph

For slide i, let the patch encoder produce n_i node features:

```math
H_i^{(0)}
=
\begin{bmatrix}
h_{i1}^{(0)\mathsf T}\\
\vdots\\
h_{in_i}^{(0)\mathsf T}
\end{bmatrix}
\in
\mathbb R^{n_i\times d_0}.
```

The feature-only MIL object is a multiset:

```math
\mathcal B_i
=
\left\{
h_{iv}^{(0)}
:
v\in V_i
\right\}.
```

A graph augments the multiset with a relation object:

```math
\mathcal G_i
=
\left(
V_i,
E_i,
A_i,
H_i^{(0)},
C_i,
E_i^{\mathrm{attr}},
T_i,
B_i
\right).
```

The components may be:

| Symbol | Role |
| --- | --- |
| V_i | patch-node index set |
| E_i | edge set |
| A_i | adjacency or weighted support |
| H_i^(0) | node-feature matrix |
| C_i | coordinate matrix |
| E_i^attr | edge features or edge weights |
| T_i | node or edge type labels |
| B_i | fine-to-coarse assignment matrix |

Not every graph MIL method uses every component. A coordinate k-nearest-neighbor
graph uses C to construct A, but may not pass C into the message function. A
heterogeneous graph may use T and edge attributes. A hierarchical model uses B
to transfer fine states to a coarser graph.

The predictor is:

```math
\widehat y_i
=
\mathcal H_\theta
\left(
\mathcal R_\theta
\left(
\mathcal C_\theta
\left(
H_i^{(0)},\mathcal G_i
\right)
\right)
\right).
```

The graph is part of context when it changes node states before readout. It is
part of geometry when its support or attributes encode a relation among
patches. The same matrix A can have both roles.

## 2. Graph construction is a modeling decision

### 2.1 Coordinate k-nearest-neighbor graph

Let c_iv be the physical coordinate of node v. A directed k-nearest-neighbor
support is:

```math
\mathcal N_i(v)
=
\mathrm{KNN}_k
\left(
c_{iv};
\{c_{iu}\}_{u\ne v}
\right).
```

A binary adjacency is:

```math
A_{i,vu}
=
\mathbf 1
\left\{
u\in\mathcal N_i(v)
\right\}.
```

A distance-weighted adjacency may be:

```math
A_{i,vu}
=
\mathbf 1
\left\{
u\in\mathcal N_i(v)
\right\}
\exp
\left(
-\frac{
\|c_{iv}-c_{iu}\|_2^2
}{
2\sigma^2
}
\right).
```

This graph encodes a proximity hypothesis. It does not prove that nearest
patch centers have the most relevant biological interaction.

### 2.2 Feature k-nearest-neighbor graph

Using a feature metric d:

```math
\mathcal N_i^{\mathrm{feat}}(v)
=
\mathrm{KNN}_k
\left(
h_{iv}^{(0)};
\{h_{iu}^{(0)}\}_{u\ne v}
\right).
```

The support then depends on the encoder:

```math
A_i
=
A_i
\left(
H_i^{(0)}
\right).
```

A feature graph can connect distant tissue regions with similar morphology. It
may capture phenotype similarity while discarding physical adjacency.

### 2.3 Learned directed graph

A learned edge score can be:

```math
r_{i,vu}
=
g_\theta
\left(
h_{iv}^{(0)},
h_{iu}^{(0)},
c_{iv},
c_{iu}
\right).
```

Top-k support is:

```math
\mathcal N_i^\theta(v)
=
\mathrm{TopK}_k
\left(
\{r_{i,vu}:u\ne v\}
\right).
```

The adjacency is now task-dependent:

```math
A_{i,vu}^\theta
=
\mathbf 1
\left\{
u\in\mathcal N_i^\theta(v)
\right\}.
```

This changes the hypothesis class and creates a discrete support-selection
boundary. A small score perturbation can switch an edge.

### 2.4 Complete graph

A complete graph has:

```math
A_{i,vu}
=
1
\quad
\text{for all }v\ne u.
```

A complete graph can emulate a set-context operator if every node receives a
shared permutation-invariant summary. It does not make every graph equivalent
to a set because edge attributes, typed relations, or multiple message-passing
steps can still carry relational information.

## 3. The permutation group action

Node order is arbitrary. Let P_i be a permutation matrix of size n_i:

```math
P_iP_i^{\mathsf T}
=
P_i^{\mathsf T}P_i
=
I_{n_i}.
```

Reindex node features by:

```math
H_i^{(0)\prime}
=
P_iH_i^{(0)}.
```

Reindex adjacency by conjugation:

```math
A_i'
=
P_iA_iP_i^{\mathsf T}.
```

For edge attributes represented as an ordered pair tensor, the corresponding
reindexing is:

```math
E_{i,vu}^{\mathrm{attr}\prime}
=
E_{i,\pi^{-1}(v)\pi^{-1}(u)}^{\mathrm{attr}}.
```

For node types:

```math
T_i'
=
P_iT_i.
```

A graph is not invariant to arbitrary changes in A. The symmetry says that
relabeling the same graph changes its matrix representation in a coordinated
way.

## 4. Equivariance and invariance

A node-context operator is permutation equivariant if:

```math
\mathcal C_\theta
\left(
P_iH_i,
P_iA_iP_i^{\mathsf T}
\right)
=
P_i
\mathcal C_\theta
\left(
H_i,A_i
\right).
```

A graph readout is permutation invariant if:

```math
\mathcal R_\theta(P_i\widetilde H_i)
=
\mathcal R_\theta(\widetilde H_i).
```

The composition is a well-defined graph-level function:

```math
f_\theta
\left(
P_iH_i,
P_iA_iP_i^{\mathsf T}
\right)
=
f_\theta(H_i,A_i).
```

The correct phrase is therefore:

```math
\text{node states are equivariant; slide representations are invariant}.
```

It is imprecise to say that the graph itself is permutation invariant. The graph
object changes coordinate representation under a node relabeling, while the
isomorphism class remains the same.

## 5. Matrix message passing

A generic message-passing layer has:

```math
m_{iv}^{(\ell)}
=
\rho_\ell
\left(
\left\{
\phi_\ell
\left(
h_{iv}^{(\ell)},
h_{iu}^{(\ell)},
e_{i,uv}
\right)
:
u\in\mathcal N_i(v)
\right\}
\right),
```

```math
h_{iv}^{(\ell+1)}
=
\zeta_\ell
\left(
h_{iv}^{(\ell)},
m_{iv}^{(\ell)}
\right).
```

The neighbor aggregator rho must be invariant to the ordering of the neighbor
multiset. Sum, mean, max, and a normalized attention sum satisfy this
condition.

In matrix form, a simple linear layer is:

```math
H_i^{(\ell+1)}
=
\varphi_\ell
\left(
\widetilde A_i
H_i^{(\ell)}
W_\ell
\right),
```

where A-tilde includes a specified self-loop and normalization convention.

For the symmetric GCN normalization:

```math
\widetilde A_i
=
D_i^{-1/2}
(A_i+I)
D_i^{-1/2},
```

```math
D_{i,vv}
=
\sum_u
(A_i+I)_{vu}.
```

Under permutation:

```math
A_i'+I
=
P_i(A_i+I)P_i^{\mathsf T},
```

```math
D_i'
=
P_iD_iP_i^{\mathsf T},
```

and therefore:

```math
\widetilde A_i'
=
P_i\widetilde A_iP_i^{\mathsf T}.
```

The transformed layer is:

```math
H_i^{(\ell+1)\prime}
=
\varphi_\ell
\left(
P_i\widetilde A_iP_i^{\mathsf T}
P_iH_i^{(\ell)}
W_\ell
\right)
=
P_i
H_i^{(\ell+1)},
```

provided the nonlinearity is applied rowwise. This proves equivariance for the
layer.

## 6. Neighbor-set proof

Let pi be the node permutation. The reindexed neighbor set is:

```math
\mathcal N_i'( \pi(v) )
=
\{\pi(u):u\in\mathcal N_i(v)\}.
```

If rho is permutation invariant:

```math
\rho
\left(
\left\{
\phi(h_{\pi(v)},h_{\pi(u)},e_{\pi(u)\pi(v)})
:
\pi(u)\in\mathcal N_i'(\pi(v))
\right\}
\right)
=
\rho
\left(
\left\{
\phi(h_v,h_u,e_{uv})
:
u\in\mathcal N_i(v)
\right\}
\right).
```

Thus the message attached to target pi-v equals the original message attached
to target v. The output rows are reindexed by P.

The proof fails if the neighbor aggregator depends on an arbitrary list order,
unless that order is itself a transformed graph attribute. It also fails if a
model silently uses node indices as positional embeddings.

## 7. Graph attention equivariance

Let the edge attention score be:

```math
e_{i,vu}
=
a_\theta
\left(
Wh_{iv},
Wh_{iu},
e_{i,vu}^{\mathrm{attr}}
\right).
```

Normalize over the incoming neighbors:

```math
\alpha_{i,vu}
=
\frac{
\exp(e_{i,vu})
}{
\sum_{r\in\mathcal N_i(v)}
\exp(e_{i,vr})
}.
```

The node update is:

```math
h_{iv}^{(\ell+1)}
=
\sigma
\left(
\sum_{u\in\mathcal N_i(v)}
\alpha_{i,vu}
Wh_{iu}^{(\ell)}
\right).
```

Under a coordinated node permutation, the terms in the denominator and sum
are reindexed, so:

```math
\alpha_{i,\pi(v)\pi(u)}'
=
\alpha_{i,vu}.
```

The output at pi-v equals the reindexed output at v. The softmax does not
destroy equivariance; it is normalized over a permuted neighbor set.

This is different from slide-level attention. Neighbor attention is a context
operator C. A global attention readout is an R operator.

## 8. Graph readouts

### 8.1 Sum

```math
z_i^{\mathrm{sum}}
=
\sum_{v\in V_i}
\widetilde h_{iv}.
```

Under permutation:

```math
\sum_v(P_i\widetilde H_i)_v
=
\sum_v\widetilde h_{iv}.
```

Sum is invariant but count-sensitive.

### 8.2 Mean

```math
z_i^{\mathrm{mean}}
=
\frac{1}{|V_i|}
\sum_{v\in V_i}
\widetilde h_{iv}.
```

Mean is invariant and normalized by node count.

### 8.3 Max

```math
z_{i,r}^{\mathrm{max}}
=
\max_{v\in V_i}
\widetilde h_{iv,r}.
```

Max is invariant but nondifferentiable at ties and retains only coordinatewise
extrema.

### 8.4 Global attention

Let q_v be a graph-readout score:

```math
q_v
=
g_\theta(\widetilde h_{iv}).
```

Normalize globally:

```math
\alpha_{iv}
=
\frac{\exp(q_v)}
{\sum_{u\in V_i}\exp(q_u)}.
```

The graph-level representation is:

```math
z_i^{\mathrm{attn}}
=
\sum_{v\in V_i}
\alpha_{iv}\widetilde h_{iv}.
```

If g is shared and rowwise, q is equivariant and z is invariant.

### 8.5 Typed readout

For node types t in a finite type set T:

```math
z_{i,t}
=
\sum_{v:T_{iv}=t}
\alpha_{iv}^{(t)}
\widetilde h_{iv}.
```

The type-specific collection can be concatenated:

```math
z_i^{\mathrm{typed}}
=
\big\Vert_{t\in\mathcal T}
z_{i,t}.
```

This preserves type-stratified first moments, but missing types require an
explicit mask or zero convention.

## 9. Readout changes the representation object

Let two contextualized node fields be U and U-prime. A readout collision is:

```math
\mathcal R(U)
=
\mathcal R(U').
```

No head after R can distinguish them from the representation alone.

Mean collision:

```math
\frac{1}{n}\sum_vu_v
=
\frac{1}{n'}\sum_vu_v'.
```

Max collision:

```math
\max_vu_{v,r}
=
\max_vu_{v,r}'
\quad
\text{for every coordinate }r.
```

Attention collision:

```math
\sum_va_vu_v
=
\sum_va_v'u_v'.
```

The graph context can make U and U-prime differ even when the raw feature
multisets match, but the final readout can still collapse the resulting fields.

## 10. Graph versus set: the exact boundary

A set model receives the multiset of node features:

```math
\mathcal B_i
=
\{h_{iv}\}_{v\in V_i}.
```

A graph model receives:

```math
(\mathcal B_i,A_i,E_i^{\mathrm{attr}},T_i).
```

If two graphs have the same feature multiset but different relation objects,
a set-only function cannot distinguish them:

```math
\{h_{iv}\}_{v\in V_i}
=
\{h_{i'v}\}_{v\in V_{i'}}
\Longrightarrow
f_{\mathrm{set}}(H_i)
=
f_{\mathrm{set}}(H_{i'}).
```

A graph function may distinguish them:

```math
f_{\mathrm{graph}}(H_i,A_i)
\ne
f_{\mathrm{graph}}(H_{i'},A_{i'}).
```

This is only possible if the graph context or graph readout actually uses A.

### Two-node counterexample

Let:

```math
H
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
```

Let the first relation connect the two nodes and the second relation use only
self-loops:

```math
\widetilde A^{\mathrm{swap}}
=
\begin{bmatrix}
0&1\\
1&0
\end{bmatrix},
\qquad
\widetilde A^{\mathrm{self}}
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}.
```

A linear context map gives:

```math
\widetilde H^{\mathrm{swap}}
=
\widetilde A^{\mathrm{swap}}H
=
\begin{bmatrix}
0\\
1
\end{bmatrix},
```

```math
\widetilde H^{\mathrm{self}}
=
\widetilde A^{\mathrm{self}}H
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
```

The raw multiset is identical. A graph-contextualized nodewise readout can
distinguish the relations; a feature-only set model cannot.

## 11. Complete graph and one-layer reduction

A complete graph does not automatically equal a set model, but under restrictive
conditions it can reduce to a shared global summary.

Suppose:

```math
m_v
=
\rho
\left(
\{h_u:u\ne v\}
\right),
```

where rho is invariant to neighbor order and all nodes have identical
self/neighbor treatment. If every node sees the same multiset up to its own
excluded feature, the layer can express a set function of the entire bag.

For example, with a sum including self:

```math
m_v
=
\sum_{u=1}^{n}h_u,
```

the same global sum is available at every node. A global readout then applies a
set function to duplicated copies of that sum.

The reduction is not universal. Edge attributes, node types, directed support,
self-loop choices, multiple layers, and node-specific features can preserve
relation information. A graph with only complete untyped support and a single
symmetric message may be set-like; an arbitrary graph is not.

## 12. Relation information and graph isomorphism

Two graph inputs are isomorphic if there is a permutation P such that:

```math
H'
=
PH,
\qquad
A'
=
PAP^{\mathsf T},
```

with corresponding edge and node attributes also reindexed.

A correctly equivariant context plus invariant readout gives the same output on
isomorphic graphs:

```math
f(H',A')
=
f(H,A).
```

Non-isomorphic graphs can also receive the same output. Invariance is required
for isomorphism, not sufficient for distinguishing every non-isomorphic graph.

Message-passing GNNs are commonly analyzed through the one-dimensional
Weisfeiler-Leman refinement. A generic color update is:

```math
c_v^{(\ell+1)}
=
\mathrm{HASH}
\left(
c_v^{(\ell)},
\{\!\{c_u^{(\ell)}:u\in\mathcal N(v)\}\!\}
\right).
```

If two graphs are indistinguishable by the corresponding refinement, a standard
message-passing architecture with matching initial features may also fail to
separate them. The graph readout cannot recover distinctions absent from the
node states.

This limitation matters in WSI graphs because a visually plausible topology
does not guarantee that the message-passing depth and readout retain the
relevant arrangement.

## 13. Depth, receptive fields, and oversmoothing

After L message-passing layers, the state at node v depends on an L-hop
neighborhood:

```math
h_v^{(L)}
=
F_L
\left(
\{h_u^{(0)}:
\mathrm{dist}_G(u,v)\le L\},
G
\right).
```

The graph-level readout then combines these local fields. Increasing L expands
the receptive field but can reduce node distinguishability.

For a normalized linear propagation:

```math
H^{(\ell+1)}
=
\widetilde A H^{(\ell)}W_\ell,
```

ignore W and nonlinearities for diagnosis:

```math
H^{(L)}
=
\widetilde A^L H^{(0)}.
```

If the graph is connected and the propagation is mixing, powers of A-tilde can
approach a low-dimensional stationary subspace:

```math
\widetilde A^L H^{(0)}
\longrightarrow
\mathbf 1\pi^{\mathsf T}H^{(0)}
\quad
\text{as }L\to\infty
+,
```

where the exact limit depends on normalization and spectral assumptions.

The practical failure is oversmoothing: node states become too similar for a
readout to distinguish local tissue patterns.

## 14. Oversquashing

A node at depth L may receive information from an exponentially growing number
of nodes through a fixed-width state. Let B_L(v) be the L-hop ball:

```math
B_L(v)
=
\{u:\mathrm{dist}_G(u,v)\le L\}.
```

The number of potential signals can grow rapidly:

```math
|B_L(v)|
\gg
d_L,
```

where d_L is the dimension of the transmitted state. Many distinct
neighborhood configurations can map to the same state. This is oversquashing.

A graph may therefore have the correct relation support but insufficient
capacity or depth to transmit a rare long-range interaction to the readout.

## 15. Geometry errors

### 15.1 Wrong k-nearest-neighbor support

If coordinate noise changes the order of neighbor distances, the graph support
changes discontinuously:

```math
\|c_v-c_a\|_2
<
\|c_v-c_b\|_2
\longrightarrow
a\in\mathcal N(v),
```

```math
\|c_v-c_a\|_2
>
\|c_v-c_b\|_2
\longrightarrow
b\in\mathcal N(v).
```

A small coordinate perturbation near a tie can change an entire message path.

### 15.2 Feature shortcut

A feature graph may connect patches by staining or scanner similarity rather
than disease-relevant morphology:

```math
A
=
A(H^{(0)})
\quad
\text{can encode nuisance variation}.
```

The model may appear relational while the relation is a shortcut.

### 15.3 Missing tissue and irregular sampling

WSI coordinates are not a regular lattice. A graph built from available patches
may have degree and density that reflect tissue segmentation, missing regions,
or magnification rather than biology.

## 16. Graph readout versus graph context

A global graph attention readout is:

```math
z_i
=
\sum_v\alpha_{iv}\widetilde h_{iv}.
```

A patch-level ABMIL readout without graph context is:

```math
z_i^{\mathrm{set}}
=
\sum_va_{iv}h_{iv}.
```

Even if alpha and a have the same formula:

```math
z_i
\ne
z_i^{\mathrm{set}}
\quad
\text{because}
\quad
\widetilde h_{iv}
=
\mathcal C(H_i,A_i)_v
\ne
h_{iv}
```

in general.

This is the graph MIL design boundary. “Attention pooling” describes R, not
the entire model.

## 17. Geometry as evidence versus geometry as prior

A coordinate graph imposes a prior:

```math
u\sim v
\quad
\text{if }
\|c_u-c_v\|
\text{ is small}.
```

Whether this relation is evidence depends on the data-generating process. If
tumor phenotype is locally coherent, it can be useful. If relevant cells are
rare and spatially separated, local smoothing can erase the signal.

A learned feature graph imposes:

```math
u\sim v
\quad
\text{if }
d(h_u,h_v)
\text{ is small}.
```

This can capture phenotype recurrence across a slide but may create a graph
shortcut based on the pretraining domain.

The graph object is thus an inductive bias, not a neutral preprocessing step.

## 18. Graph-level task heads

For classification:

```math
\widehat y_i
=
\sigma(w^{\mathsf T}z_i+b).
```

For a Cox risk head:

```math
r_i
=
w^{\mathsf T}z_i+b.
```

The partial likelihood uses the graph-derived risk:

```math
\mathcal L_{\mathrm{Cox}}
=
-
\sum_{i:\delta_i=1}
\left[
r_i
-
\log
\sum_{j\in\mathcal R(T_i)}
\exp(r_j)
\right].
```

The graph context and readout are the same representation pathway, but the
supervision changes the target direction and therefore the useful graph
statistics.

For a horizon-specific survival head:

```math
\widehat h_i^{(k)}
=
\sigma
\left(
w_k^{\mathsf T}z_i+b_k
\right).
```

An explanation of attention weights for classification does not automatically
explain a Cox risk or a hazard at horizon k.

## 19. C/R/G/S placement

| Component | Graph MIL meaning |
| --- | --- |
| C | message passing, graph convolution, graph attention, typed relation, or hierarchical transfer |
| R | sum, mean, max, global attention, typed pooling, or graph-level set readout |
| G | coordinate support, feature similarity, learned directed edges, edge attributes, or assignment hierarchy |
| S | slide classification, survival, subtype, localization, or auxiliary node/edge targets |

The full map is:

```math
H_i^{(0)}
\xrightarrow{
\mathcal G_i
}
\mathcal C_{\theta}
\left(
H_i^{(0)},\mathcal G_i
\right)
=
\widetilde H_i
\xrightarrow{
\mathcal R_{\theta}
}
z_i
\xrightarrow{
\mathcal H_{\theta}
}
\widehat y_i.
```

A graph readout over raw features is not equivalent to a graph readout over
contextualized features. A coordinate visualization is not equivalent to a
coordinate-conditioned predictor.

## 20. Sanity checks

### 20.1 Node permutation

Sample a permutation P and compare:

```math
f(H,A)
\quad
\text{with}
\quad
f(PH,PAP^{\mathsf T}).
```

They should agree up to numerical tolerance.

### 20.2 Feature-only versus relation intervention

Hold H fixed and replace A:

```math
f(H,A)
\quad
\text{versus}
\quad
f(H,A').
```

If the output never changes, the graph relation may not be used. If it changes
dramatically under tiny support perturbations, topology sensitivity may be too
high.

### 20.3 Complete-graph reduction

Compare a graph layer on complete support with a set-context baseline. The
outputs should only match under the same self-loop, normalization, edge-feature,
and readout conventions.

### 20.4 Depth sweep

Measure node-state pairwise similarity:

```math
S^{(\ell)}
=
\frac{1}{n(n-1)}
\sum_{u\ne v}
\cos
\left(
h_u^{(\ell)},h_v^{(\ell)}
\right).
```

A monotone rise toward one can diagnose oversmoothing.

### 20.5 Topology stability

Perturb coordinates or feature distances and measure edge-set change:

```math
\mathrm{Jaccard}(E,E')
=
\frac{|E\cap E'|}{|E\cup E'|}.
```

A low score under negligible perturbation indicates a brittle graph constructor.

### 20.6 Readout collision

Construct two node fields with equal mean but different tails. If the graph
readout gives the same representation, the tail is not surviving R.

## 21. Bottom line

Graph MIL is a function on a relational object:

```math
f_\theta:
(H_i,A_i,E_i^{\mathrm{attr}},T_i,C_i)
\longmapsto
\widehat y_i.
```

Its correctness requires equivariant node context and invariant slide readout:

```math
\mathcal C(PH,PAP^{\mathsf T})
=
P\mathcal C(H,A),
\qquad
\mathcal R(P\widetilde H)
=
\mathcal R(\widetilde H).
```

Its scientific meaning depends on the graph construction:

```math
\text{graph inductive bias}
=
\text{which patch relations are allowed to exchange information}.
```

A complete graph can approximate set context under restrictive symmetry, but a
general graph is not a set with extra bookkeeping. The adjacency, edge
attributes, node types, and assignment maps can change the representation
object before aggregation.

The C/R/G/S summary is:

```math
\boxed{
\text{graph MIL}
=
\text{equivariant relational context}
+
\text{invariant graph readout}
+
\text{explicit graph geometry}
+
\text{task supervision}
}
```

The decisive question is:

```math
\text{which graph relations survive the context and readout, and which are lost?}
```
