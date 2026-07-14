# Graph MIL Failure Modes and Counterexamples

Graph MIL adds a relation object before the slide-level bottleneck. That extra
object can preserve spatial or semantic interactions, but it can also create
new shortcuts and new information loss. A useful failure analysis therefore
has to identify the operator that is responsible.

For a slide, write the complete predictor as:

```math
G_i
=
\Gamma_{\mathrm{geom}}
\left(
H_i^{(0)},C_i,\tau_i,\phi_i
\right),
```

```math
\widetilde H_i
=
\mathcal C_{\theta}
\left(
H_i^{(0)},G_i
\right),
```

```math
z_i
=
\mathcal R_{\theta}
\left(
\widetilde H_i,G_i
\right),
```

```math
\widehat y_i
=
\mathcal H_{\theta}(z_i).
```

Here:

- Gamma_geom constructs the graph object;
- C is graph context;
- R is graph readout;
- H is the task head;
- tau and phi are optional node and edge attributes.

A graph failure is not one undifferentiated phenomenon. It can occur because
the graph is wrong, because context compresses the wrong neighborhood, because
the readout discards the relevant statistic, or because supervision rewards a
shortcut.

## 1. A four-way failure decomposition

Let the desired slide-relevant relation be G-star and the implemented relation
be G. Let the desired contextual node field be H-star and the implemented field
be H-tilde. Define:

```math
\begin{aligned}
\text{geometry error}
&=
d_{\mathrm G}(G,G^\star),
\\
\text{context error}
&=
d_{\mathrm H}
\left(
\mathcal C_{\theta}(H^{(0)},G),
\mathcal C^\star(H^{(0)},G^\star)
\right),
\\
\text{readout collision}
&=
\mathcal R_{\theta}(\widetilde H)
-
\mathcal R_{\theta}(\widetilde H'),
\\
\text{supervision shortcut}
&=
\text{predictive association with }G
\text{ that is not stable under deployment}.
\end{aligned}
```

The first three are representation failures. The fourth is a statistical
failure: the training objective can prefer a graph artifact even when the
forward map is implemented correctly.

The C/R/G/S placement is:

```math
\boxed{
\text{G chooses relations}
+
\text{C propagates them}
+
\text{R compresses the resulting field}
+
\text{S decides which collisions are acceptable}
}
```

## Source anchors

The paper-specific examples in this note are anchored to the graph MIL notes
that precede it:

| Method | Source note | Role in this failure analysis |
| --- | --- | --- |
| Patch-GCN | [coordinate context and survival](02_patch_gcn_coordinate_context_and_survival.md) | coordinate kNN, DeepGCN-style context, multiscale states, global attention |
| HEAT | [typed context and pseudo-label readout](03_heat_typed_context_and_pseudo_label_readout.md) | feature kNN, Pearson edge attributes, teacher types, PL-Pool |
| WiKG | [dynamic support and knowledge readout](04_wikg_dynamic_support_and_knowledge_readout.md) | directed head-tail support, relation embeddings, selected-support attention |
| HACT | [hierarchy as graph MIL](05_hact_hierarchy_as_graph_mil.md) | cell graph, tissue graph, and hard assignment transfer |

The failure arguments below are general mathematical consequences of those
objects. They should not be read as claims that every implementation exhibits
every failure mode.

## 2. Wrong support is a representation error

Consider a one-layer linearized message-passing map:

```math
H^{(1)}
=
\widetilde A H^{(0)}W,
```

where A-tilde includes self-loops, edge weights, and normalization. If the
relevant relation is A-star but the model receives A, then:

```math
\Delta H^{(1)}
=
\left(
\widetilde A-\widetilde A^\star
\right)
H^{(0)}W.
```

A norm bound is:

```math
\left\|
\Delta H^{(1)}
\right\|_{\mathrm F}
\le
\left\|
\widetilde A-\widetilde A^\star
\right\|_2
\left\|
H^{(0)}
\right\|_{\mathrm F}
\left\|
W
\right\|_2.
```

This bound says that a perfect patch encoder does not compensate for a
relation operator that is far from the task-relevant one. Increasing depth
propagates the discrepancy through more layers; it does not identify the
missing relation automatically.

The paper-specific support choices are different:

```text
Patch-GCN:
    approximate coordinate k-nearest-neighbor support

HEAT:
    feature-space k-nearest-neighbor support

WiKG:
    directed top-k support selected from head-tail compatibility

HACT:
    cell graph, tissue graph, and a hard cell-to-tissue assignment
```

They should not be compared as if they differ only in message-passing layer.

## 3. Exact two-node graph counterexample

Use identical node features in two slides:

```math
H^{(0)}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
```

Let the first graph use identity propagation and the second graph use a
fully mixing self-loop-normalized support:

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
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\widetilde A_2H^{(0)}
=
\begin{bmatrix}
1/2\\
1/2
\end{bmatrix}.
```

A graph context can therefore distinguish two slides whose patch multiset is
identical. This is the intended advantage when the relation is meaningful.
It is also the exact mechanism by which a graph artifact becomes predictive.

If the final readout is a mean, the two graph-level values are:

```math
\frac{1}{2}
\mathbf 1^{\mathsf T}
\widetilde A_1H^{(0)}
=
\frac{1}{2},
\qquad
\frac{1}{2}
\mathbf 1^{\mathsf T}
\widetilde A_2H^{(0)}
=
\frac{1}{2}.
```

In this particular example, graph context changes node states but the mean
readout cancels the difference. A max readout gives:

```math
\max_v
\left[
\widetilde A_1H^{(0)}
\right]_v
=
1,
\qquad
\max_v
\left[
\widetilde A_2H^{(0)}
\right]_v
=
1/2.
```

The same context error can therefore be visible or invisible depending on R.

## 4. Coordinate graph mismatch

Patch-GCN builds support from physical patch coordinates. Let p-v be the
coordinate of node v and define:

```math
\mathcal N_k(v)
=
\mathrm{TopK}_{u\ne v}
\left(
-\left\|p_v-p_u\right\|_2
\right).
```

This geometry is useful when local spatial neighborhoods approximate tissue
interactions. It fails when the relevant evidence is separated by a gap, a
mask, a fold, or a tissue boundary.

Let two positive regions be far apart in coordinate space but semantically
linked. A k-nearest graph with insufficient k omits their relation:

```math
u\notin\mathcal N_k(v)
\qquad
\text{even though}
\qquad
I(u;Y\mid X_v)>0.
```

The context map cannot pass that interaction in one layer. With L layers, it
can only arrive if a path exists:

```math
u\in B_L(v)
\quad\Longrightarrow\quad
\text{a path of length at most L connects u and v}.
```

A coordinate graph can also create false neighbors across a tissue boundary:

```math
\left\|p_u-p_v\right\|_2
\text{ small}
\qquad
\not\Longrightarrow
\qquad
\text{same biological compartment}.
```

A coordinate perturbation test should move a patch across the boundary while
holding its embedding fixed. If the output changes sharply, the model is using
the graph construction as a biological boundary prior.

## 5. Feature graph mismatch

HEAT constructs feature-space k-nearest-neighbor support. Let the encoder
feature be x-v and define:

```math
\mathcal N_k^{\mathrm{feat}}(v)
=
\mathrm{TopK}_{u\ne v}
\mathrm{sim}(x_v,x_u).
```

This can connect distant patches with similar staining or texture. The support
then represents encoder similarity rather than physical adjacency:

```math
\left\|x_u-x_v\right\|_2
\text{ small}
\qquad
\not\Longrightarrow
\qquad
\left\|p_u-p_v\right\|_2
\text{ small}.
```

That may be desirable for semantic retrieval, but it changes the claim from
local spatial context to feature-mediated context.

For a graph layer using support N(v), the difference between spatial and feature
context is:

```math
\Delta h_v
=
\Phi_{\theta}
\left(
h_v,
\{h_u:u\in\mathcal N_k^{\mathrm{feat}}(v)\}
\right)
-
\Phi_{\theta}
\left(
h_v,
\{h_u:u\in\mathcal N_k^{\mathrm{coord}}(v)\}
\right).
```

The two supports should be evaluated separately. A single heatmap cannot show
which geometry generated the message.

## 6. Learned support and top-k discontinuity

For WiKG, let s-v-u be the head-tail compatibility score from target v to
candidate u. The selected support is:

```math
\mathcal N_K(v)
=
\mathrm{TopK}_{u\ne v}
s_{vu}.
```

Let s-v-(K) and s-v-(K+1) denote the Kth and K+1st ordered scores. Define the
margin:

```math
\gamma_v
=
s_{v,(K)}
-
s_{v,(K+1)}.
```

If gamma-v is positive, sufficiently small score perturbations preserve the
support. If gamma-v is zero, an arbitrarily small perturbation can replace an
edge:

```math
\gamma_v=0
\Longrightarrow
\mathcal N_K(v)
\text{ is not locally unique}.
```

Within a fixed support, selected softmax weights have the usual derivative:

```math
\pi_{vu}
=
\frac{\exp(q_{vu})}
{\sum_{r\in\mathcal N_K(v)}\exp(q_{vr})},
```

```math
\frac{\partial\pi_{vu}}{\partial q_{vr}}
=
\pi_{vu}
\left(
\mathbf 1\{u=r\}
-
\pi_{vr}
\right).
```

At a support boundary, this derivative describes only the continuous
conditional map. It does not describe the discrete edge replacement.

A useful instability statistic is:

```math
\kappa_v
=
\frac{1}{|\mathcal N_K(v)|}
\sum_{u\in\mathcal N_K(v)}
\mathbf 1
\left\{
u\text{ changes under a small perturbation}
\right\}.
```

Report support stability separately from attention-weight stability.

## 7. Directed-role failure in WiKG

WiKG assigns head and tail representations:

```math
h_v=W_hx_v,
\qquad
t_u=W_tx_u.
```

The candidate relation is directional:

```math
s_{vu}
=
h_v^{\mathsf T}t_u.
```

In general:

```math
s_{vu}
\ne
s_{uv}.
```

A silent symmetrization replaces the intended object with:

```math
s_{vu}^{\mathrm{sym}}
=
\frac{1}{2}
\left(
s_{vu}+s_{uv}
\right).
```

This removes direction-specific compatibility. Two directed graphs can have the
same undirected support but different ordered triples:

```math
(v,r_{vu},u)
\ne
(u,r_{uv},v).
```

The relation embedding in the implementation is constructed from the selected
head and tail states:

```math
r_{vu}
=
\widetilde\omega_{vu}t_u
+
\left(
1-\widetilde\omega_{vu}
\right)h_v.
```

If the relation state is dropped, the downstream message cannot distinguish
two edges with identical tail features but different head-conditioned relation
states. The failure is not ordinary oversmoothing; it is loss of ordered edge
information.

## 8. Edge-attribute failure in HEAT

HEAT uses node types and Pearson endpoint-feature correlation as edge data. Let
the edge attribute be:

```math
\phi_{vu}
=
\mathrm{corr}(x_v,x_u).
```

The edge-conditioned attention score can be written schematically as:

```math
e_{vu}
=
\frac{
\left(
W_K^{\tau(u)}x_u
\odot
W_E\phi_{vu}
\right)^{\mathsf T}
W_Q^{\tau(v)}x_v
}{
\sqrt d
}.
```

An edge attribute perturbation gives:

```math
\Delta e_{vu}
=
\frac{
\left(
W_Kx_u
\odot
W_E(\phi_{vu}+\Delta\phi_{vu})
\right)^{\mathsf T}
W_Qx_v
-
e_{vu}\sqrt d
}{
\sqrt d
}.
```

If the projected edge vector is large, small correlation errors can alter the
attention ranking. The endpoint correlation is not a neutral annotation; it is
inside C.

A permutation test that independently shuffles edge attributes while keeping
node features and support fixed measures how much the model relies on phi.
A type-shuffle test measures the corresponding dependence on tau.

## 9. Oversmoothing as spectral contraction

For a symmetric normalized propagation operator A-tilde, ignore feature
transforms and write:

```math
H^{(L)}
=
\widetilde A^L H^{(0)}.
```

Let the eigendecomposition be:

```math
\widetilde A
=
U\Lambda U^{\mathsf T},
\qquad
\Lambda
=
\mathrm{diag}
(\lambda_1,\ldots,\lambda_N).
```

Assume lambda-1 equals one and all other eigenvalues have magnitude strictly
less than one. Then:

```math
\widetilde A^L
=
U\Lambda^LU^{\mathsf T}
\longrightarrow
u_1u_1^{\mathsf T}
```

when the dominant eigenspace is one-dimensional. Therefore:

```math
H^{(L)}
\longrightarrow
u_1u_1^{\mathsf T}H^{(0)}.
```

For a connected regular graph, u-1 is proportional to the all-ones vector, so
node states approach a common value. The contraction rate is controlled by:

```math
\rho_{\mathrm{smooth}}
=
\max_{r\ge2}|\lambda_r|.
```

The residual difference after L layers is bounded by:

```math
\left\|
\left(
I-u_1u_1^{\mathsf T}
\right)
H^{(L)}
\right\|_{\mathrm F}
\le
\rho_{\mathrm{smooth}}^L
\left\|
\left(
I-u_1u_1^{\mathsf T}
\right)
H^{(0)}
\right\|_{\mathrm F}.
```

Residual and dense connections can preserve shallow representations, as in
Patch-GCN, but they do not make a single deep propagation operator
non-contracting. The relevant diagnostic is the layerwise representation
distance:

```math
D_\ell
=
\frac{1}{|E|}
\sum_{(u,v)\in E}
\left\|
h_u^{(\ell)}-h_v^{(\ell)}
\right\|_2.
```

If D-ell collapses while the task score remains stable, the model may be using
a coarse graph statistic or a shortcut rather than preserving local evidence.

## 10. Oversquashing as a Jacobian bottleneck

Let B-L(v) be the L-hop receptive field of node v. The input dimension of that
neighborhood is approximately:

```math
m_v
=
|B_L(v)|d_{\mathrm{in}}.
```

The node state has dimension d. When m-v greatly exceeds d, the map:

```math
\Psi_v:
\mathbb R^{m_v}
\longrightarrow
\mathbb R^d
```

must compress many configurations. Its Jacobian has rank at most d:

```math
\mathrm{rank}
J_{\Psi_v}
\le
d.
```

Thus any perturbation in the local nullspace satisfies:

```math
J_{\Psi_v}\Delta x=0
\quad
\Longrightarrow
\quad
\text{no first-order change in the node state}.
```

For a graph readout, a distant source u has influence:

```math
\frac{\partial z}{\partial x_u}
=
\frac{\partial z}{\partial \widetilde H}
\frac{\partial \widetilde H}{\partial x_u}.
```

The second factor contains all message paths from u to the nodes that R sees.
If paths are numerous but each carries a small Jacobian factor, the product
can become numerically small. This is a quantitative version of long-range
dependence loss.

A topology test should compare the shortest path length and the total gradient
norm:

```math
\ell(u,v)
=
\mathrm{dist}_G(u,v),
\qquad
I_{uv}
=
\left\|
\frac{\partial \widetilde h_v}{\partial x_u}
\right\|_{\mathrm F}.
```

A decreasing I-u-v with increasing distance does not by itself prove
oversquashing, but a sharp decrease after a support or degree change is
diagnostic.

## 11. Depth, diameter, and disconnected components

A message-passing stack of L layers cannot transmit information across graph
distance greater than L unless the architecture includes a global operation.
For a disconnected graph:

```math
G
=
G_1\sqcup\cdots\sqcup G_c,
```

local context preserves component separation:

```math
\widetilde H_{G_j}
=
\mathcal C_j(H_{G_j},G_j).
```

A sum or mean readout can still combine components, but it cannot create
messages between them. If a biological relation crosses components, the model
must encode it through a global readout, a hierarchy, or an additional edge
family.

The component-size bias differs by readout. For component representations
z-j:

```math
z^{\mathrm{sum}}
=
\sum_{j=1}^{c}z_j^{\mathrm{sum}},
```

```math
z^{\mathrm{mean}}
=
\frac{
\sum_{j=1}^{c}n_jz_j^{\mathrm{mean}}
}{
\sum_{j=1}^{c}n_j
}.
```

A mean over all nodes weights large components more heavily than small
components. A mean of component means would define a different statistic.

## 12. Size and density shortcuts

Suppose every contextualized node equals u. Sum and mean give:

```math
z^{\mathrm{sum}}=Nu,
\qquad
z^{\mathrm{mean}}=u.
```

If node count N correlates with the label because slides have different tissue
area, then a sum model can perform well without learning morphology. If N is
correlated with the label only because of tissue segmentation quality, the
shortcut will fail under a preprocessing change.

Let q be a scalar task score. A simple decomposition is:

```math
q
=
a^{\mathsf T}z^{\mathrm{sum}}
+
bN
+
c.
```

If z-sum is proportional to N, the two terms are confounded. A count-residual
test fits:

```math
q_{\mathrm{res}}
=
q-\widehat{\mathbb E}[q\mid N].
```

A large performance drop after residualizing or matching N across groups
indicates count dependence. This does not make count useless; it identifies
whether count is an explicit or accidental statistic.

## 13. Degree normalization is another statistic

For a graph with adjacency A and degree matrix D, common propagation uses:

```math
\widetilde A
=
D^{-1/2}(A+I)D^{-1/2}.
```

The self-loop contribution at node v is:

```math
\widetilde A_{vv}
=
\frac{1}{d_v+1}.
```

Higher degree reduces the coefficient on the node's own feature. A degree change
can therefore alter a node state even when all neighbor features are unchanged:

```math
\Delta h_v
=
\left(
\frac{1}{d_v'+1}
-
\frac{1}{d_v+1}
\right)h_v
+
\text{neighbor-term changes}.
```

Degree is not merely a normalization detail. It is a graph-level covariate that
can encode tissue density, masking, or patch sampling.

## 14. Pseudo-type errors in HEAT

Let tau-v be the true semantic type and tau-hat-v be the teacher-generated
type. The type error indicator is:

```math
\varepsilon_v
=
\mathbf 1\{\widehat\tau_v\ne\tau_v\}.
```

A type error can affect three separate paths:

```math
\widehat\tau_v
\longrightarrow
\text{type-specific Q/K/V projections},
```

```math
\widehat\tau_v
\longrightarrow
\text{edge-type interaction},
```

```math
\widehat\tau_v
\longrightarrow
\text{type-indexed PL row}.
```

The error is amplified when the projection matrices for two types differ
strongly:

```math
\left\|
W_Q^{a}-W_Q^{b}
\right\|_2
+
\left\|
W_K^{a}-W_K^{b}
\right\|_2
+
\left\|
W_V^{a}-W_V^{b}
\right\|_2
\text{ is large}.
```

For a type row containing m-a nodes, the row mean has variance:

```math
\mathrm{Var}
\left(
\frac{1}{m_a}
\sum_{v:\widehat\tau_v=a}
\widetilde h_v
\right)
=
\frac{1}{m_a^2}
\sum_{v:\widehat\tau_v=a}
\mathrm{Var}(\widetilde h_v)
+
\text{covariance terms}.
```

Rare types therefore have high estimation variance even when the teacher is
accurate.

## 15. HACT assignment nullspace

Let B be the cell-to-tissue assignment matrix with shape:

```math
B\in\{0,1\}^{N_{\mathrm C}\times N_{\mathrm T}},
\qquad
\sum_{b=1}^{N_{\mathrm T}}B_{ab}=1.
```

The transfer of contextualized cell states is:

```math
U_{\mathrm C\to T}
=
B^{\mathsf T}\widetilde H_{\mathrm C}.
```

Any fine-scale perturbation Delta satisfying:

```math
B^{\mathsf T}\Delta=0
```

is invisible to this transfer:

```math
B^{\mathsf T}
\left(
\widetilde H_{\mathrm C}+\Delta
\right)
=
B^{\mathsf T}\widetilde H_{\mathrm C}.
```

For a tissue-level function that depends only on the transfer and fixed tissue
features:

```math
F_{\mathrm T}
\left(
H_{\mathrm T},
B^{\mathsf T}\widetilde H_{\mathrm C}
\right),
```

the two cell fields collide:

```math
F_{\mathrm T}
\left(
H_{\mathrm T},
B^{\mathsf T}(\widetilde H_{\mathrm C}+\Delta)
\right)
=
F_{\mathrm T}
\left(
H_{\mathrm T},
B^{\mathsf T}\widetilde H_{\mathrm C}
\right).
```

This is not a generic claim that all HACT implementations lose every
within-region contrast. A later cell-level skip connection or auxiliary loss
could preserve extra information. The nullspace statement applies to the
assignment transfer path itself.

## 16. Hierarchy errors are not ordinary graph noise

If a cell is assigned to the wrong tissue region, the error changes the
coarse-scale membership:

```math
B_{ab}=1
\longrightarrow
B_{ab'}=1,
\qquad
b'\ne b.
```

The transfer perturbation is:

```math
\Delta U
=
\left(
e_{b'}-e_b
\right)
\widetilde h_{\mathrm C,a}^{\mathsf T},
```

where e-b is the bth standard basis vector in tissue-node space. A single
assignment error moves the complete cell state from one tissue row to another.

If the tissue readout is a sum, the total transferred mass is unchanged:

```math
\mathbf 1^{\mathsf T}U_{\mathrm C\to T}
=
\mathbf 1^{\mathsf T}B^{\mathsf T}\widetilde H_{\mathrm C}.
```

But the distribution of mass across tissue regions changes. A later tissue graph
can amplify that relocation because it propagates the error to neighboring
regions.

## 17. Graph isomorphism and message-passing collisions

A standard message-passing layer has the form:

```math
h_v^{(\ell+1)}
=
\Phi_{\ell}
\left(
h_v^{(\ell)},
\mathrm{AGG}_{u\in\mathcal N(v)}
\Psi_{\ell}
\left(
h_v^{(\ell)},h_u^{(\ell)},e_{vu}
\right)
\right).
```

When the aggregation is permutation invariant and parameters are shared, the
architecture is bounded by the one-dimensional Weisfeiler-Lehman refinement
in its ability to distinguish certain graph structures.

Two regular graphs can have identical initial node features and identical
degree-refinement colors. In that setting, every node can receive the same
state at every layer:

```math
h_v^{(\ell)}
=
h_u^{(\ell)}
\qquad
\forall u,v.
```

Any permutation-invariant readout then produces the same representation for
graphs with the same number of nodes, even if their cycle structure differs.
Adding coordinates, edge attributes, node types, or positional encodings can
break this collision, but it also changes the graph object and its inductive
bias.

The practical claim is limited:

```math
\text{message passing plus readout}
\ne
\text{complete graph isomorphism test}.
```

This is relevant when a pathology label depends on a global motif that is not
recoverable by the chosen local refinement.

## 18. Readout collisions after correct context

Even a perfect graph context can be followed by a lossy R. For contextualized
fields H-tilde and H-tilde-prime:

```math
\mathcal R(\widetilde H)
=
\mathcal R(\widetilde H')
\Longrightarrow
\mathcal H(\mathcal R(\widetilde H))
=
\mathcal H(\mathcal R(\widetilde H')).
```

Mean collision:

```math
\frac{1}{N}\sum_v\widetilde h_v
=
\frac{1}{N'}\sum_u\widetilde h_u'.
```

Max collision:

```math
\max_v[\widetilde h_v]_r
=
\max_u[\widetilde h_u']_r
\qquad
\forall r.
```

Attention collision:

```math
\sum_v\alpha_v\widetilde h_v
=
\sum_u\alpha_u'\widetilde h_u'.
```

Typed collision:

```math
\mathcal R_a(\widetilde H_a)
=
\mathcal R_a(\widetilde H_a')
\qquad
\forall a.
```

The graph can be constructed correctly and still fail because the chosen
surviving statistic does not match the label-generating mechanism.

## 19. Attention weight is not deletion effect

For global graph attention:

```math
z
=
\sum_v\alpha_v\widetilde h_v,
\qquad
\alpha_v
=
\frac{\exp(s_v)}{\sum_u\exp(s_u)}.
```

If the task head is locally linear:

```math
q
=
w^{\mathsf T}z+b,
```

the signed additive term is:

```math
r_v
=
\alpha_vw^{\mathsf T}\widetilde h_v.
```

The exact fixed-state deletion effect is:

```math
q-q_{-v}
=
\frac{\alpha_v}{1-\alpha_v}
w^{\mathsf T}
\left(
\widetilde h_v-z_{-v}
\right).
```

The full graph deletion effect is instead:

```math
\Delta_v^{\mathrm{full}}
=
q(G)-q(G\setminus v).
```

The full effect includes changes to messages, support, degrees, type rows, and
normalization. It is not equal to alpha-v, r-v, or fixed-state deletion in
general.

## 20. Fixed-graph versus rebuilt-graph deletion

Define four interventions:

```math
\begin{aligned}
q_{\mathrm{full}}
&=
q(G,H),
\\
q_{\mathrm{node}}
&=
q(G,H\text{ with node }v\text{ removed}),
\\
q_{\mathrm{edge}}
&=
q(G\setminus E_v,H\text{ with node }v\text{ retained}),
\\
q_{\mathrm{rebuild}}
&=
q(\Gamma_{\mathrm{geom}}(H\text{ with node }v\text{ removed})).
\end{aligned}
```

Their differences separate effects:

```math
\Delta_{\mathrm{node}}
=
q_{\mathrm{full}}-q_{\mathrm{node}}
```

measures feature and readout removal under fixed support.

```math
\Delta_{\mathrm{context}}
=
q_{\mathrm{node}}-q_{\mathrm{edge}}
```

measures the effect of removing incident messages under a controlled node
state convention.

```math
\Delta_{\mathrm{geometry}}
=
q_{\mathrm{edge}}-q_{\mathrm{rebuild}}
```

measures the effect of rebuilding the graph after deletion. The exact
definitions depend on whether the implementation permits an isolated node,
but the principle is that an explanation must name the intervention.

## 21. Graph-shift failure at deployment

Let P-train and P-test be distributions over graph objects and labels. A model
trained on graph geometry can fail even when the patch feature distribution is
stable:

```math
P_{\mathrm{train}}(H)
\approx
P_{\mathrm{test}}(H),
\qquad
P_{\mathrm{train}}(G\mid H)
\ne
P_{\mathrm{test}}(G\mid H).
```

Examples include:

```text
different scanner magnification or coordinate normalization
different tissue segmentation and patch filtering
different k-nearest-neighbor implementation
different feature encoder or stain preprocessing
different nucleus detector and pseudo-type calibration
different tissue-region segmentation for HACT
```

A graph-specific shift audit compares, on matched node identities or a stated
common alignment:

```math
\begin{aligned}
\Delta_{\mathrm{degree}}
&=
d\left(P_{\mathrm{train}}(\deg G),P_{\mathrm{test}}(\deg G)\right),
\\
\Delta_{\mathrm{support}}
&=
\mathbb E
\left[
1-
\frac{|E_{\mathrm{train}}\cap E_{\mathrm{test}}|}
{|E_{\mathrm{train}}\cup E_{\mathrm{test}}|}
\right],
\\
\Delta_{\mathrm{type}}
&=
d\left(P_{\mathrm{train}}(\tau),P_{\mathrm{test}}(\tau)\right).
\end{aligned}
```

A high graph shift with stable node embeddings is evidence that the geometry
operator, not the patch encoder, may be the deployment bottleneck.

## 22. Patient leakage through graph construction

If multiple slides or patches from one patient enter both training and
validation, graph construction can leak patient-specific structure even when
the label is not copied directly. Let P be patient identity. The desired split
condition is:

```math
P_{\mathrm{train}}
\cap
P_{\mathrm{test}}
=
\varnothing.
```

For a patient-level graph, all derived objects must follow the split:

```math
\Gamma_{\mathrm{geom}}(H_i),
\quad
\tau_i,
\quad
\phi_i,
\quad
\text{and any graph memory bank}.
```

A graph built from a global feature kNN index can also leak test-slide features
into the training graph if the index is fit before splitting. The correct
construction is:

```math
\Gamma_{\mathrm{train}}
=
\Gamma(H_{\mathrm{train}}),
\qquad
\Gamma_{\mathrm{test}}
=
\Gamma(H_{\mathrm{test}})
```

unless cross-slide retrieval is an explicitly defined and leakage-controlled
part of the model.

## 23. Counterexample: same mean, different topology

Let all initial node features be scalar one. Consider two four-node graphs:
a cycle and two disjoint edges. A one-layer degree-normalized mean context gives
the same node state on both graphs because every node has the same degree under
the chosen self-loop normalization:

```math
\widetilde A_{\mathrm{cycle}}\mathbf 1
=
\mathbf 1,
\qquad
\widetilde A_{\mathrm{two\ edges}}\mathbf 1
=
\mathbf 1.
```

A mean readout therefore collides:

```math
\frac{1}{4}
\mathbf 1^{\mathsf T}
\widetilde A_{\mathrm{cycle}}\mathbf 1
=
\frac{1}{4}
\mathbf 1^{\mathsf T}
\widetilde A_{\mathrm{two\ edges}}\mathbf 1
=
1.
```

The graphs have different connectivity and cycle structure. The model cannot
recover that distinction from constant node features and this readout. Edge
attributes or nonconstant node features are required to make the topology
observable.

## 24. Counterexample: same node multiset, different graph risk

Let:

```math
H^{(0)}
=
\begin{bmatrix}
1\\
1\\
0\\
0
\end{bmatrix}.
```

Let G-one connect the two positive nodes and the two negative nodes, while
G-two connects each positive node to a negative node. With a neighbor-mean
context:

```math
h_v^{(1)}
=
\frac{1}{d_v}
\sum_{u\in\mathcal N(v)}h_u^{(0)},
```

the positive nodes receive:

```math
\begin{aligned}
G_1:
&\quad
h_{\mathrm{positive}}^{(1)}=1,
\\
G_2:
&\quad
h_{\mathrm{positive}}^{(1)}=0.
\end{aligned}
```

A max readout can distinguish the graphs:

```math
z_{\mathrm{max}}(G_1)=1,
\qquad
z_{\mathrm{max}}(G_2)=0.
```

The patch multiset is identical. The graph relation alone changes the
surviving witness.

## 25. Counterexample: mean versus existence

Let Z-v be a latent positive-instance indicator and suppose the target is:

```math
Y_{\mathrm{exist}}
=
\mathbf 1
\left\{
\sum_{v=1}^{N}Z_v>0
\right\}.
```

For two slides with positive fractions 1/N and 1/(2N), the mean statistic is
different but both labels are one. A finite-dimensional head can learn the
threshold only if N is available or the graph context produces a separable
witness.

For a burden target:

```math
Y_{\mathrm{burden}}
=
\mathbf 1
\left\{
\frac{1}{N}\sum_{v=1}^{N}Z_v>\tau
\right\},
```

the mean is aligned with the target while max is blind to multiplicity. Thus
the readout failure depends on the supervision semantics, not on an abstract
ranking of pooling methods.

## 26. Counterexample: typed versus untyped pooling

Let two node types a and b have scalar contextual states:

```math
G_1:
\quad
H_a=\{1\},
\quad
H_b=\{0\},
```

```math
G_2:
\quad
H_a=\{0\},
\quad
H_b=\{1\}.
```

An untyped mean collides:

```math
\frac{1+0}{2}
=
\frac{0+1}{2}.
```

A typed row readout distinguishes them:

```math
Z(G_1)
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
Z(G_2)
=
\begin{bmatrix}
0\\
1
\end{bmatrix}.
```

HEAT's fixed semantic row coordinate preserves a distinction that ordinary
pooling erases. If the teacher types are wrong, that same coordinate system
can encode the wrong distinction.

## 27. Counterexample: hierarchy transfer collision

Take two cells assigned to one tissue node and scalar cell states:

```math
H_{\mathrm C}
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
H_{\mathrm C}'
=
\begin{bmatrix}
1/2\\
1/2
\end{bmatrix},
\qquad
B
=
\begin{bmatrix}
1\\
1
\end{bmatrix}.
```

Then:

```math
B^{\mathsf T}H_{\mathrm C}
=
B^{\mathsf T}H_{\mathrm C}'
=
1.
```

A tissue graph receiving only the assignment sum cannot distinguish a
concentrated cell signal from a distributed one. A cell-level max, variance,
or auxiliary readout would be required to preserve that distinction.

## 28. Sanity-check suite

### 28.1 Feature permutation

Permute node order and transform all graph-indexed objects consistently:

```math
H'
=
PH,
\qquad
A'
=
PAP^{\mathsf T}.
```

A correctly implemented invariant graph predictor should satisfy:

```math
F(H',A',\tau',\phi')
=
F(H,A,\tau,\phi).
```

### 28.2 Geometry intervention

Hold H fixed and compare two supports:

```math
F(H,A)
\ne
F(H,A')
```

only if the context or readout genuinely uses the changed relation. The
difference should be measured, not inferred from a diagram.

### 28.3 Readout isolation

Compute contextualized node states once, then compare:

```math
z^{\mathrm{sum}},
\qquad
z^{\mathrm{mean}},
\qquad
z^{\mathrm{max}},
\qquad
z^{\mathrm{attn}}.
```

This isolates R from C.

### 28.4 Support stability

Perturb features or parameters by epsilon and measure edge Jaccard overlap:

```math
J(E,E')
=
\frac{|E\cap E'|}{|E\cup E'|}.
```

Report J as a function of epsilon and the top-k margin.

### 28.5 Message sensitivity

For each edge, remove only that edge while holding the node set fixed:

```math
\Delta_{uv}^{\mathrm{edge}}
=
F(G)-F(G\setminus\{(u,v)\}).
```

This is distinct from deleting a node or rebuilding support.

### 28.6 HACT assignment nullspace

Construct a perturbation Delta with zero sum inside every tissue region:

```math
B^{\mathsf T}\Delta=0.
```

The tissue transfer should remain unchanged. If it does not, an additional
cell-level path is active and should be documented.

### 28.7 HEAT type shuffle

Shuffle pseudo-types while preserving their marginal counts:

```math
\tau'
=
\mathrm{permute}(\tau).
```

Compare task output, type rows, and edge-conditioned messages. This tests
semantic-type dependence separately from feature-support dependence.

### 28.8 WiKG rank perturbation

Perturb a score near the Kth boundary and record support replacement:

```math
\gamma_v
\approx
0
\Longrightarrow
J(E,E')
\text{ should be monitored}.
```

### 28.9 Graph-shift audit

Evaluate by graph-density, type-frequency, and support-overlap strata, not only
by a single slide-level average.

## 29. Paper-specific failure matrix

| Paper | G failure | C failure | R failure | S or deployment risk |
| --- | --- | --- | --- | --- |
| Patch-GCN | coordinate kNN misses or invents physical neighbors | smoothing, finite receptive field, oversquashing | global attention loses multiplicity and topology | slide area, segmentation, and cohort geometry shift |
| HEAT | feature kNN is not spatial adjacency | type-specific and edge-conditioned message is teacher-sensitive | PL rows erase within-type arrangement | pseudo-type calibration and rare-type variance |
| WiKG | top-k directed support is discontinuous | head-tail roles and relation states can encode shortcuts | mean/max erase selected edge triples | support instability and feature-distribution shift |
| HACT | cell-region assignment is wrong or incomplete | cell and tissue contexts amplify parent errors | coarse transfer has assignment nullspace | segmentation quality and region-scale shift |
| generic graph MIL | graph isomorphism or topology collisions | oversmoothing and oversquashing | statistic mismatch at global readout | loss rewards graph artifacts |

## 30. C/R/G/S failure checklist

Before claiming that a graph MIL model captures pathology context, record:

```math
\begin{aligned}
G &: \text{how is support constructed and what does an edge mean?}
\\
C &: \text{what messages are exchanged and what is their receptive field?}
\\
R &: \text{which statistic survives and what collisions does it create?}
\\
S &: \text{which labels, sampling rules, and losses reward the representation?}
\end{aligned}
```

The minimum intervention set is:

```text
support permutation or replacement
node-feature permutation
edge deletion
node deletion with fixed support
node deletion with rebuilt support
readout swap
type or assignment shuffle
count and density matching
patient-level graph split
```

Without these checks, a performance gain cannot be attributed cleanly to graph
context.

## 31. Bottom line

Graph MIL has two bottlenecks:

```math
\boxed{
\text{graph construction}
\longrightarrow
\text{message passing}
\longrightarrow
\text{graph readout}
}
```

The first bottleneck decides which relations can exist. The second decides
which information about those relations reaches the task head.

The strongest general counterexample is:

```math
G\ne G',
\qquad
\{x_v\}_{v\in V}
=
\{x_v'\}_{v\in V'},
\qquad
\mathcal R(\mathcal C(G))
\ne
\mathcal R(\mathcal C(G')).
```

This shows the intended power of graph MIL. The corresponding failure is:

```math
\mathcal R(\mathcal C(G))
=
\mathcal R(\mathcal C(G'))
\qquad
\text{for task-relevant }G\ne G'.
```

The correct conclusion is not that graph context is universally better than set
MIL. It is:

```math
\boxed{
\text{graph context is useful only when its relation is valid, its bottlenecks
retain the task statistic, and its supervision does not reward unstable shortcuts}
}
```
