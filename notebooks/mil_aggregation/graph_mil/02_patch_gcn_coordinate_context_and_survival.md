# Patch-GCN: Coordinate Context, Hierarchical Message Passing, and Survival Readout

Source:

Chen et al., Whole Slide Images are 2D Point Clouds: Context-Aware Survival
Prediction using Patch-based Graph Convolutional Networks, MICCAI 2021.
[arXiv](https://arxiv.org/abs/2107.13048)

The paper's central move is to treat a whole slide as a two-dimensional point
cloud rather than as an unordered collection of isolated patch embeddings. It
constructs a graph from physical patch coordinates, propagates local context
with DeepGCN-style layers, and applies global attention pooling to the
contextualized node field before survival supervision.

The forward map is:

```math
\text{patch images}
\longrightarrow
\text{coordinate graph}
\longrightarrow
\text{graph-contextualized nodes}
\longrightarrow
\text{WSI embedding}
\longrightarrow
\text{patient risk}.
```

The model is not simply “GCN plus survival.” Its inductive bias is:

```math
\boxed{
\text{nearby tissue patches should exchange information before global pooling}
}
```

## 1. Patient-level object

The survival observation belongs to a patient:

```math
O_i
=
(T_i,\delta_i),
```

where T-i is the observed time and delta-i is the event indicator. If patient i
has K-i WSIs, write:

```math
P_i
=
\{W_{ij}\}_{j=1}^{K_i}.
```

Patch-GCN constructs a graph for each WSI:

```math
G_{ij}
=
\left(
V_{ij},
E_{ij},
X_{ij},
C_{ij},
A_{ij}
\right).
```

The patient-level graph object is therefore a collection:

```math
\mathcal G_i
=
\{G_{ij}\}_{j=1}^{K_i}.
```

This distinction matters. The graph convolution is within a WSI graph. The
survival label is patient-level. There is no need to invent edges between
separate slides from the same patient when the paper does not specify them.

For one WSI, suppress the patient and slide indices:

```math
G
=
(V,E,X,C,A).
```

The node count is M, the number of extracted patches in that WSI.

## 2. Patch preprocessing and feature object

The paper uses non-overlapping tissue patches at 20x magnification:

```math
x_m
\in
\mathbb R^{256\times256\times3},
\qquad
m\in\{1,\ldots,M\}.
```

Tissue localization is performed before patch extraction. The paper describes
an Otsu-based tissue segmentation pipeline on a downsampled slide; the
important representation consequence is that the graph contains tissue
patches, not every pixel or every possible grid location.

A truncated ResNet-50 produces a fixed patch embedding:

```math
h_m^{(0)}
=
f_{\mathrm{ResNet50}}
\left(
x_m
\right)
\in
\mathbb R^{1024}.
```

The paper specifies spatial average pooling after the third residual block for
the patch representation. Stack the features:

```math
X
=
\begin{bmatrix}
h_1^{(0)\mathsf T}\\
\vdots\\
h_M^{(0)\mathsf T}
\end{bmatrix}
\in
\mathbb R^{M\times1024}.
```

The encoder is a feature map from image appearance to node attributes. It does
not yet communicate information between patches:

```math
h_m^{(0)}
=
f(x_m)
\quad
\text{depends on }x_m\text{ but not on }x_u\text{ for }u\ne m.
```

The graph context operator begins only after this feature extraction step.

## 3. Coordinate object

Each extracted patch has a physical slide coordinate:

```math
c_m
=
\begin{bmatrix}
x_m^{\mathrm{coord}}\\
y_m^{\mathrm{coord}}
\end{bmatrix}
\in
\mathbb R^2.
```

Stack the coordinates:

```math
C
=
\begin{bmatrix}
c_1^{\mathsf T}\\
\vdots\\
c_M^{\mathsf T}
\end{bmatrix}
\in
\mathbb R^{M\times2}.
```

The coordinate is not merely used for plotting. It defines the graph support.
The graph's geometry is therefore coordinate geometry rather than feature
similarity geometry:

```math
A
=
\Gamma_{\mathrm{coord}}(C).
```

A feature graph would instead be:

```math
A_{\mathrm{feat}}
=
\Gamma_{\mathrm{feat}}(X).
```

Patch-GCN uses the first construction. This is an explicit inductive bias.

## 4. Coordinate kNN graph

For node v, compute Euclidean coordinate distance:

```math
d_{vu}
=
\|c_v-c_u\|_2.
```

The approximate coordinate k-nearest-neighbor set is:

```math
\mathcal N(v)
=
\mathrm{KNN}_{8}
\left(
c_v;
\{c_u\}_{u\ne v}
\right).
```

The paper uses k equal to 8. A directed row-target convention is:

```math
A_{vu}
=
\mathbf 1
\left\{
u\in\mathcal N(v)
\right\}.
```

The corresponding edge set is:

```math
E
=
\left\{
(v,u):
A_{vu}=1
\right\}.
```

The graph object is:

```math
G
=
(X,A,C).
```

The paper emphasizes fast approximate coordinate kNN rather than a fully
specified graph-theoretic convention for every implementation detail. The
following choices should therefore be reported if used, but should not be
retroactively attributed to the paper:

```text
radius cutoff
self-loop convention
edge symmetrization
directed versus undirected storage
cross-WSI edges
edge feature construction
```

The faithful paper-level statement is:

```math
\text{support}
=
\text{approximate coordinate kNN with }k=8.
```

## 5. Why k equals 8 is a geometry assumption

On a regular patch lattice, the central patch and its eight surrounding
neighbors form a 3-by-3 local field. Patch-GCN uses k equal to 8 as a graph
analogue of that local receptive field.

The graph does not guarantee an exact 3-by-3 lattice because tissue masks can
remove patches and slide boundaries can create irregular degrees. Let:

```math
\deg(v)
=
|\mathcal N(v)|.
```

Even if the target k is 8, the effective local geometry can be irregular when
patches are missing or when approximate kNN ties are resolved differently.

The local-support assumption is:

```math
u\in\mathcal N(v)
\Longrightarrow
\text{patch }u\text{ is a useful candidate source of context for }v.
```

The converse is also important:

```math
u\notin\mathcal N(v)
\Longrightarrow
\text{there is no direct one-layer message }u\to v.
```

A biologically relevant interaction at a larger distance must be represented
through multiple graph hops or a later global readout.

## 6. Coordinate geometry versus feature geometry

Coordinate support:

```math
\mathcal N_{\mathrm{coord}}(v)
=
\mathrm{KNN}_{8}
\left(
c_v;C
\right).
```

Feature support:

```math
\mathcal N_{\mathrm{feat}}(v)
=
\mathrm{KNN}_{k}
\left(
h_v^{(0)};X
\right).
```

These encode different hypotheses:

```math
\mathcal N_{\mathrm{coord}}:
\text{nearby tissue should interact},
```

```math
\mathcal N_{\mathrm{feat}}:
\text{similar-looking tissue should interact}.
```

Two patches can be spatial neighbors with very different embeddings. Patch-GCN
allows their features to exchange messages anyway. Two distant patches with
similar morphology are not directly connected by the coordinate graph.

This is why Patch-GCN is a geometry-aware context model rather than a
feature-similarity graph MIL model.

## 7. General DeepGCN-style context layer

At layer ell, node v has state:

```math
h_v^{(\ell)}
\in
\mathbb R^{d_\ell}.
```

A generic message-passing layer is:

```math
m_{vu}^{(\ell)}
=
\phi_\ell
\left(
h_v^{(\ell)},
h_u^{(\ell)},
e_{vu}^{(\ell)}
\right),
\qquad
u\in\mathcal N(v).
```

The node message is aggregated by a neighbor-set operator:

```math
m_v^{(\ell)}
=
\rho_\ell
\left(
\{m_{vu}^{(\ell)}:u\in\mathcal N(v)\}
\right).
```

The state update is:

```math
h_v^{(\ell+1)}
=
\zeta_\ell
\left(
h_v^{(\ell)},
m_v^{(\ell)}
\right).
```

Patch-GCN adapts a DeepGCN-style message construction and softmax aggregation.
The WSI-specific geometry enters through N-v; the message formula is applied
on that coordinate support.

## 8. Patch-GCN message construction

The generic edge-aware message used in the source formulation can be written:

```math
m_{vu}^{(\ell)}
=
\mathrm{ReLU}
\left(
h_u^{(\ell)}
+
\mathbf 1
\left\{
h_{e,vu}^{(\ell)}\ne0
\right\}
h_{e,vu}^{(\ell)}
\right)
+
\varepsilon,
```

with:

```math
\varepsilon
=
10^{-7}.
```

The edge state h-e-v-u is included in the general edge-aware expression. The
paper's WSI graph is primarily specified by coordinate support. It should not be
described as using a biological edge attribute unless such an attribute is
explicitly supplied by the implementation.

A dimension-safe abstraction is:

```math
m_{vu}^{(\ell)}
=
\phi_\ell
\left(
h_u^{(\ell)},
e_{vu}^{(\ell)}
\right).
```

The important source-specific ingredients are:

```text
ReLU message construction
small positive numerical offset
neighbor aggregation on coordinate kNN support
residual and dense graph-depth connections
```

## 9. Featurewise softmax aggregation

The source writes a softmax aggregation over graph messages. For a feature
coordinate r, define:

```math
a_{vu,r}^{(\ell)}
=
\frac{
\exp
\left(
m_{vu,r}^{(\ell)}
\right)
}{
\sum_{s\in\mathcal N(v)}
\exp
\left(
m_{vs,r}^{(\ell)}
\right)
}.
```

The aggregated message coordinate is:

```math
\bar m_{v,r}^{(\ell)}
=
\sum_{u\in\mathcal N(v)}
a_{vu,r}^{(\ell)}
m_{vu,r}^{(\ell)}.
```

Vector notation is:

```math
\bar m_v^{(\ell)}
=
\left[
\bar m_{v,1}^{(\ell)},
\ldots,
\bar m_{v,d_\ell}^{(\ell)}
\right].
```

The local coefficients satisfy:

```math
\sum_{u\in\mathcal N(v)}
a_{vu,r}^{(\ell)}
=
1
\quad
\text{for every feature coordinate }r.
```

This is not the same object as the global scalar attention coefficient used
later by MIL pooling. The local coefficient has indices v, u, r, and ell. The
global readout coefficient has one node index and normalizes over all WSI
nodes.

The local operator retains a featurewise weighted first moment of neighbor
messages:

```math
\bar m_{v,r}^{(\ell)}
=
\mathbb E_{a_{v,\cdot,r}^{(\ell)}}
\left[
m_{vu,r}^{(\ell)}
\right].
```

The graph-contextualized state then contains local relational information.

## 10. Temperature and numerical stability

A temperature-parameterized local softmax would be:

```math
a_{vu,r}^{(\ell,\tau)}
=
\frac{
\exp
\left(
m_{vu,r}^{(\ell)}/\tau
\right)
}{
\sum_{s\in\mathcal N(v)}
\exp
\left(
m_{vs,r}^{(\ell)}/\tau
\right)
}.
```

The source expression corresponds to tau equal to one. As tau tends to zero,
the local operator approaches a coordinatewise maximum over neighbor messages.
As tau tends to infinity, it approaches a coordinatewise mean.

For numerical stability, subtract the neighborhood maximum:

```math
a_{vu,r}^{(\ell)}
=
\frac{
\exp
\left(
m_{vu,r}^{(\ell)}
-
\max_{s\in\mathcal N(v)}
m_{vs,r}^{(\ell)}
\right)
}{
\sum_{t\in\mathcal N(v)}
\exp
\left(
m_{vt,r}^{(\ell)}
-
\max_{s\in\mathcal N(v)}
m_{vs,r}^{(\ell)}
\right)
}.
```

This algebraically leaves the coefficient unchanged while avoiding overflow.

## 11. Node update and residual path

The local update is:

```math
h_v^{(\ell+1)}
=
\mathrm{MLP}_\ell
\left(
h_v^{(\ell)}
+
\bar m_v^{(\ell)}
\right).
```

A residual graph block can be represented as:

```math
H^{(\ell+1)}
=
H^{(\ell)}
+
\mathcal F_\ell
\left(
H^{(\ell)},A
\right).
```

At node v:

```math
h_v^{(\ell+1)}
=
h_v^{(\ell)}
+
f_\ell
\left(
h_v^{(\ell)},
\{\bar m_{vu}^{(\ell)}\}_{u\in\mathcal N(v)}
\right).
```

The residual path provides a direct route for the original representation. If
the graph message is uninformative or harmful, the optimizer can in principle
reduce the residual correction. This does not guarantee protection from
oversmoothing, but it changes the failure mode relative to pure repeated
averaging.

## 12. Dense multi-depth representation

Patch-GCN uses dense multi-depth graph connections. A compact notation is:

```math
\widetilde h_v
=
\mathrm{Concat}
\left(
h_v^{(1)},
h_v^{(2)},
h_v^{(3)},
h_v^{(4)}
\right).
```

The exact implementation may also include a projection or input path. The
paper-level point is that multiple graph depths are retained rather than only
the final state.

The dimensions are:

```math
h_v^{(\ell)}
\in
\mathbb R^{d_\ell},
\qquad
\widetilde h_v
\in
\mathbb R^{d_1+d_2+d_3+d_4}
```

before any final projection.

The surviving node statistic is therefore multiscale:

```math
\widetilde h_v
=
\left(
\text{shallow local context},
\text{deeper context},
\text{four-hop context}
\right).
```

This is the sense in which Patch-GCN hierarchically aggregates local and
global topological structure. It is not a cell-to-region hierarchy with
explicit parent assignments.

## 13. Four graph layers and receptive field

The source uses four graph convolution layers:

```math
L=4.
```

Define the L-hop ball:

```math
\mathcal B_L(v)
=
\left\{
u:
\mathrm{dist}_G(u,v)\le L
\right\}.
```

Ignoring skip connections, the final node state can depend on:

```math
\widetilde h_v
=
F_\theta
\left(
\{h_u^{(0)}:u\in\mathcal B_4(v)\},
G[\mathcal B_4(v)]
\right).
```

The source reports an effective image receptive field of approximately:

```math
2302
\times
2302.
```

That value should be treated as paper-reported. A naive regular-grid
calculation can differ by boundary and indexing conventions. The safe
mathematical statement is that graph depth expands the receptive field from a
single patch to a four-hop coordinate neighborhood.

## 14. Local graph attention is not global MIL attention

The local featurewise coefficient is:

```math
a_{vu,r}^{(\ell)}
=
\frac{
\exp(m_{vu,r}^{(\ell)})
}{
\sum_{s\in\mathcal N(v)}
\exp(m_{vs,r}^{(\ell)})
}.
```

The global coefficient is:

```math
b_v
=
\frac{
\exp(s_v)
}{
\sum_{u=1}^{M}\exp(s_u)
}.
```

The index sets differ:

```math
a_{vu,r}^{(\ell)}
:
u\in\mathcal N(v),
```

```math
b_v
:
v\in V.
```

Their meanings differ:

```text
a:
    which neighboring messages shape node v in feature coordinate r

b:
    which contextualized node states shape the WSI embedding
```

A Patch-GCN heatmap based on b-v is not a visualization of the local message
coefficients.

## 15. Global attention readout

Let the contextualized node matrix be:

```math
\widetilde H
=
\begin{bmatrix}
\widetilde h_1^{\mathsf T}\\
\vdots\\
\widetilde h_M^{\mathsf T}
\end{bmatrix}
\in
\mathbb R^{M\times d_{\mathrm{out}}}.
```

The global attention score is:

```math
s_v
=
g_\theta
\left(
\widetilde h_v
\right).
```

The normalized WSI attention is:

```math
b_v
=
\frac{
\exp(s_v)
}{
\sum_{u=1}^{M}
\exp(s_u)
}.
```

The WSI representation is:

```math
z
=
\sum_{v=1}^{M}
b_v\widetilde h_v
\in
\mathbb R^{d_{\mathrm{out}}}.
```

The readout is a weighted first moment of graph-contextualized states:

```math
z
=
\mathbb E_{\widehat\mu_{\mathrm{graph}}}
\left[
\widetilde H
\right],
```

where the learned empirical measure has masses b-v on graph nodes.

The important distinction is:

```math
z
\ne
\sum_vb_vh_v^{(0)}
\quad
\text{in general}.
```

The value vectors have already absorbed spatial context through C.

## 16. Global attention deletion identity

If global attention is recomputed after deleting node j, define:

```math
w_v
=
\exp(s_v),
\qquad
S
=
\sum_{u=1}^{M}w_u,
\qquad
A
=
\sum_{u=1}^{M}w_u\widetilde h_u.
```

Then:

```math
z
=
\frac{A}{S}.
```

After deleting j while keeping the remaining contextualized node states fixed:

```math
z_{-j}
=
\frac{
A-w_j\widetilde h_j
}{
S-w_j
}.
```

The representation deletion effect is:

```math
z-z_{-j}
=
\frac{b_j}{1-b_j}
\left(
\widetilde h_j-z_{-j}
\right).
```

But in the full Patch-GCN intervention, deleting a node may also alter graph
neighborhoods and every downstream node state. The full deletion is:

```math
\Delta_j^{(q)}
=
q(X,A)
-
q(X_{-j},A_{-j}),
```

not merely the global-pooling identity above.

The identity is a readout diagnostic. It should not be presented as a complete
causal effect of a patch on a graph-contextualized survival prediction.

## 17. Patient and multi-WSI readout

For patient i and slide j:

```math
z_{ij}
=
\mathcal R_{\mathrm{attn}}
\left(
\mathcal C_{\mathrm{PatchGCN}}
\left(
G_{ij}
\right)
\right).
```

If there are multiple WSIs, the patient-level representation is some map:

```math
z_i^{\mathrm{patient}}
=
\mathcal R_{\mathrm{patient}}
\left(
\{z_{ij}\}_{j=1}^{K_i}
\right).
```

The patient risk is:

```math
\eta_i
=
\mathcal H_{\mathrm{surv}}
\left(
z_i^{\mathrm{patient}}
\right).
```

The source-specific paper object is a collection of WSI graphs per patient.
The exact cross-WSI readout should be stated according to the implementation;
it should not be replaced by an invented inter-slide graph.

If there is one WSI per patient, the map reduces to:

```math
\eta_i
=
w^{\mathsf T}z_i+b.
```

## 18. Cox-style risk head

A scalar proportional-hazards risk score is:

```math
\eta_i
=
w^{\mathsf T}z_i+b.
```

The Cox model writes the conditional hazard as:

```math
\lambda(t\mid z_i)
=
\lambda_0(t)\exp(\eta_i).
```

For event set E and risk set R-i, the negative partial log-likelihood with
Breslow-style no-ties notation is:

```math
\mathcal L_{\mathrm{Cox}}
=
-
\sum_{i:\delta_i=1}
\left[
\eta_i
-
\log
\left(
\sum_{j\in\mathcal R(T_i)}
\exp(\eta_j)
\right)
\right].
```

A bias term b cancels from each Cox risk-set term if it is shared across
patients:

```math
\eta_i
=
w^{\mathsf T}z_i+b
\Longrightarrow
\eta_i-\log\sum_{j\in\mathcal R(T_i)}e^{\eta_j}
\text{ is independent of }b.
```

This is why the graph representation, not an absolute calibrated baseline
hazard, is central to the Cox objective.

If ties exist, a tie correction such as Efron or a specified Breslow convention
is required. The paper-level note should not imply that the neural graph
architecture resolves statistical tie handling.

## 19. Survival score versus global attention

The global attention coefficient b-v is not the survival risk contribution by
itself. For a locally linear head:

```math
\eta
=
w^{\mathsf T}z+b_0
=
\sum_v
b_v
w^{\mathsf T}\widetilde h_v
+
b_0.
```

The signed additive logit term is:

```math
r_v
=
b_v
w^{\mathsf T}\widetilde h_v.
```

The coefficient b-v is nonnegative, while r-v can be negative. A node can
receive high attention while opposing the survival-risk direction.

Under full graph context, deleting v changes other node states and b weights.
Therefore:

```math
r_v
\ne
\eta(X,A)-\eta(X_{-v},A_{-v})
\quad
\text{in general}.
```

A survival heatmap should say whether it displays b-v, r-v, gradient credit,
or graph-aware deletion.

## 20. Coordinate support and permutation equivariance

Let P be a node permutation:

```math
X'
=
PX,
\qquad
C'
=
PC,
\qquad
A'
=
PAP^{\mathsf T}.
```

Coordinate kNN construction is equivariant if ties are resolved in a
permutation-consistent way:

```math
A'
=
\Gamma_{\mathrm{coord}}(C')
=
P\Gamma_{\mathrm{coord}}(C)P^{\mathsf T}.
```

The graph context satisfies:

```math
\mathcal C(PX,PAP^{\mathsf T})
=
P\mathcal C(X,A).
```

Global attention readout is invariant:

```math
\mathcal R(P\widetilde H)
=
\mathcal R(\widetilde H).
```

Thus the slide representation is invariant to node storage order while still
being sensitive to the relation encoded by A.

A coordinate permutation that changes C without conjugating A is not a relabeling
of the same graph. It is a different geometry intervention.

## 21. Local support as a Markov blanket approximation

One graph layer exposes only one-hop context:

```math
h_v^{(1)}
=
F
\left(
h_v^{(0)},
\{h_u^{(0)}:u\in\mathcal N(v)\}
\right).
```

Four layers expose a four-hop approximation:

```math
h_v^{(4)}
=
F
\left(
\{h_u^{(0)}:u\in\mathcal B_4(v)\},
G[\mathcal B_4(v)]
\right).
```

This can be interpreted as a learned local Markov blanket only if the graph
support is a reasonable proxy for conditional dependence. Coordinate
proximity does not guarantee conditional independence outside the neighborhood.

A long-range interaction not connected by a short path can be lost or
compressed before the global readout sees it.

## 22. Oversmoothing analysis

Consider a simplified linear propagation:

```math
H^{(\ell+1)}
=
\widetilde A H^{(\ell)}W_\ell.
```

Ignoring W, repeated propagation is:

```math
H^{(L)}
=
\widetilde A^L H^{(0)}.
```

If A-tilde is a mixing operator, its non-leading spectral components decay:

```math
\widetilde A^L
=
\lambda_1^L u_1v_1^{\mathsf T}
+
\sum_{r\ge2}
\lambda_r^L u_rv_r^{\mathsf T}.
```

When the magnitudes of lambda-r for r greater than one are below one, the
corresponding variation shrinks with depth.

Residual and dense connections change the propagation from a pure power to a
multi-scale combination:

```math
\widetilde h_v
=
\mathrm{Concat}
\left(
h_v^{(1)},\ldots,h_v^{(4)}
\right).
```

They reduce the information loss of using only the deepest state, but do not
make the graph immune to smoothing.

## 23. Oversquashing analysis

Let B-l-v be the l-hop neighborhood. A fixed-width node vector must encode all
signals in that set:

```math
\{h_u^{(0)}:u\in\mathcal B_\ell(v)\}
\longrightarrow
h_v^{(\ell)}\in\mathbb R^{d_\ell}.
```

If the neighborhood grows faster than the representation width, many distinct
configurations collide:

```math
|\mathcal B_\ell(v)|
\gg
d_\ell.
```

A rare distant interaction may have a negligible influence on h-v even if a
path exists. Global attention cannot recover information already compressed
away by the node context map.

## 24. Topology perturbation failure

Suppose nodes a and b are nearly tied in coordinate distance from v:

```math
d_{va}
=
d_{vb}
+
\epsilon.
```

A perturbation smaller than epsilon can reverse the kNN order:

```math
d_{va}'
>
d_{vb}'
\Longrightarrow
a\notin\mathcal N(v),
\qquad
b\in\mathcal N(v).
```

The adjacency changes discretely:

```math
A'
\ne
A.
```

The node output can then change through every message path downstream of v.
Coordinate kNN is therefore locally discontinuous at distance ties.

## 25. Missing tissue and density bias

The coordinate graph is built on observed tissue patches. Let q-v be the
probability that a potential patch at v is retained by tissue segmentation.
The observed degree is:

```math
\deg_{\mathrm{obs}}(v)
=
|\mathcal N_{\mathrm{obs}}(v)|.
```

It need not equal the degree in an unobserved complete tissue lattice. Sparse
tissue, folds, holes, and segmentation errors can change the local graph
geometry.

A graph layer can then learn a mixture of:

```text
morphology
tissue density
segmentation artifacts
slide preparation
```

The survival head cannot identify which source created a predictive state from
patient-level labels alone.

## 26. Spatial adjacency is not biological interaction

The coordinate graph encodes:

```math
u\sim v
\quad
\text{if }
c_u
\text{ and }
c_v
\text{ are nearby}.
```

It does not encode:

```math
u\sim_{\mathrm{bio}}v
\quad
\text{if }
u
\text{ and }
v
\text{ participate in a causal tissue interaction}.
```

The model assumes that local coordinate neighborhoods are a useful basis for
learning the latter. This may be reasonable for cell-tissue morphology but
should be tested with topology ablations and long-range counterexamples.

## 27. Global attention over contextualized nodes

A high global weight b-v means:

```math
\text{node }v\text{ has high routing mass in the final WSI embedding}.
```

It does not mean:

```math
\Pr
\left(
Z_v=1
\mid
X,Y
\right)
=
b_v.
```

Nor does it mean:

```math
\Delta_v^{(\eta)}
=
\eta(X,A)-\eta(X_{-v},A_{-v})
\text{ is large}.
```

The contextualized value h-tilde-v can contain messages from neighboring patches,
and deleting v can alter the neighboring states. Any final explanation must
specify the target and whether graph recomputation is performed.

## 28. Complexity

Let d be the hidden width and k equal to 8. A sparse graph layer has message
complexity approximately:

```math
\mathcal O(Mkd^2)
```

for dense feature transforms, with storage:

```math
\mathcal O(Mk+Md).
```

Global attention has:

```math
\mathcal O(Md)
```

score and weighted-sum cost, up to the score-network cost.

The full pipeline scales linearly in M for fixed k and fixed depth L:

```math
\mathcal O(LMkd^2+Md).
```

This is the computational argument for sparse coordinate support. A dense
all-pairs context operator would have quadratic edge count:

```math
\mathcal O(M^2d).
```

The sparse graph is efficient, but the efficiency comes with a restricted
support and finite receptive field.

## 29. C/R/G/S placement

| Component | Patch-GCN instantiation |
| --- | --- |
| C | DeepGCN-style ReLU message, featurewise local softmax aggregation, MLP update, residual and dense multi-depth connections |
| R | global scalar attention weighted first moment over contextualized nodes |
| G | approximate coordinate kNN graph with k equal to 8 |
| S | patient-level survival time, event indicator, and Cox-style risk supervision |

The composition is:

```math
G_{ij}
=
(X_{ij},A_{ij},C_{ij})
\xrightarrow{\mathcal C_{\mathrm{PatchGCN}}}
\widetilde H_{ij}
\xrightarrow{\mathcal R_{\mathrm{global\ attention}}}
z_{ij}
\xrightarrow{\mathcal H_{\mathrm{surv}}}
\eta_i.
```

For multiple WSIs per patient:

```math
\eta_i
=
\mathcal H_{\mathrm{surv}}
\left(
\mathcal R_{\mathrm{patient}}
\left(
\{z_{ij}\}_{j=1}^{K_i}
\right)
\right).
```

The graph geometry appears before the MIL readout. This is why a global
attention map in Patch-GCN is a map over contextualized spatial nodes rather
than a map over independent patch embeddings.

## 30. Sanity checks

### 30.1 Coordinate versus feature graph

Hold X fixed and construct A-coordinate and A-feature. Compare:

```math
\mathcal C(X,A_{\mathrm{coord}})
\quad
\text{and}
\quad
\mathcal C(X,A_{\mathrm{feat}}).
```

If they are identical, the implementation may not be using the graph support as
intended.

### 30.2 Node permutation

Sample P and compare:

```math
f(X,A)
\quad
\text{with}
\quad
f(PX,PAP^{\mathsf T}).
```

The predictions should agree up to numerical tolerance.

### 30.3 Edge deletion

Remove one local edge and measure:

```math
\Delta_{vu}
=
\eta(X,A)-\eta(X,A-\{(v,u)\}).
```

This probes topology sensitivity without deleting a node feature.

### 30.4 Patch deletion

Delete a node and recompute the graph:

```math
\Delta_v
=
\eta(X,A)-\eta(X_{-v},A_{-v}).
```

Compare this with the global attention b-v and the signed logit term:

```math
b_v,
\qquad
b_vw^{\mathsf T}\widetilde h_v,
\qquad
\Delta_v.
```

They need not rank patches identically.

### 30.5 Depth ablation

Compare representations after each depth:

```math
z^{(1)},z^{(2)},z^{(3)},z^{(4)}.
```

A monotone rise in node similarity with a fall in survival ranking can diagnose
oversmoothing.

### 30.6 k ablation

Compare coordinate supports with k equal to 4, 8, and 16. This tests whether
the result depends on a narrow local topology or a broader neighborhood.

### 30.7 Empty or sparse neighborhood

Verify the behavior when a node has fewer than k available neighbors due to
tissue masking. The implementation must specify padding, self-loops, or a
variable-degree aggregation convention.

## 31. Bottom line

Patch-GCN can be written as:

```math
\boxed{
\eta_i
=
\mathcal H_{\mathrm{surv}}
\left(
\mathcal R_{\mathrm{global}}
\left(
\mathcal C_{\mathrm{DeepGCN}}
\left(
X_i,
\Gamma_{\mathrm{coord}}(C_i)
\right)
\right)
\right)
}
```

The paper-specific choices are:

```text
20x non-overlapping 256-by-256 tissue patches
1024-dimensional truncated ResNet-50 patch features
approximate coordinate kNN with k equal to 8
DeepGCN-style local softmax message aggregation
four graph convolution layers
residual and dense multi-depth node representation
global attention pooling
patient-level survival supervision
```

The surviving statistic is a global attention-weighted first moment of
multi-scale, coordinate-contextualized node states. The main failure modes are
wrong support, graph smoothing, finite receptive field, density bias, and the
misinterpretation of global attention as causal or probabilistic patch
importance.

The C/R/G/S summary is:

```math
\boxed{
\text{coordinate context}
+
\text{multiscale graph states}
+
\text{global attention readout}
+
\text{survival risk head}
}
```

The correct question is not whether the graph “understands” the slide. It is:

```math
\text{which coordinate-neighborhood interactions survive four graph layers and
one global weighted first moment?}
```
