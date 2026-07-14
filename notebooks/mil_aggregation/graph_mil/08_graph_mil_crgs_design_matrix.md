# Graph MIL C/R/G/S Design Matrix

This note turns graph MIL into an explicit design language. The four letters are
not labels attached after the fact:

```math
\boxed{
\text{G determines the relation object}
\longrightarrow
\text{C propagates relation-conditioned context}
\longrightarrow
\text{R compresses the contextual field}
\longrightarrow
\text{S selects useful invariances and collisions}
}
```

A paper can use the same readout as another paper and still define a different
slide representation because its graph object or context operator is different.
Conversely, two graph architectures can have nearly identical graph-level
representations if their contextualized fields are mapped to the same surviving
statistic.

The decomposition is therefore a comparison tool, not a claim that the four
operators are statistically independent.

## 1. The graph MIL object

For slide i, let the initial node field be:

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

Let optional geometry, node attributes, and edge attributes be:

```math
P_i,
\qquad
\tau_i:V_i\to\mathcal A,
\qquad
\phi_i:E_i\to\mathcal E.
```

The graph construction operator is:

```math
G_i
=
\Gamma_{\mathrm G}
\left(
H_i^{(0)},P_i,\tau_i,\phi_i
\right).
```

The graph object can be represented as:

```math
G_i
=
\left(
V_i,E_i,A_i,X_i,\tau_i,\phi_i
\right).
```

A context operator maps the graph object to a contextualized node field:

```math
\widetilde H_i
=
\mathcal C_{\theta}
\left(
H_i^{(0)},G_i
\right)
\in
\mathbb R^{n_i\times d}.
```

A readout maps the field and any explicitly retained graph attributes to a
slide vector:

```math
z_i
=
\mathcal R_{\theta}
\left(
\widetilde H_i,G_i
\right)
\in
\mathbb R^q.
```

Finally, the task head is:

```math
\widehat y_i
=
\mathcal H_{\theta}(z_i).
```

A full graph MIL predictor is therefore:

```math
F_{\theta}
=
\mathcal H_{\theta}
\circ
\mathcal R_{\theta}
\circ
\mathcal C_{\theta}
\circ
\Gamma_{\mathrm G}.
```

## 2. What C/R/G/S mean

### Geometry G

G answers:

```math
\text{which pairs, types, scales, or relations are available to context?}
```

Examples:

```text
coordinate k-nearest-neighbor support
feature-space k-nearest-neighbor support
directed head-tail top-k support
cell graph plus tissue graph plus assignment matrix
```

### Context C

C answers:

```math
\text{how can an available relation alter a node state?}
```

Examples include local message passing, edge-conditioned attention, directed
knowledge-aware interaction, and two-level cell-to-tissue propagation.

### Readout R

R answers:

```math
\text{which statistic of the contextualized field reaches the task head?}
```

Examples include sum, mean, max, global attention, type-indexed rows, and
hierarchical transfer followed by coarse pooling.

### Supervision S

S is the task-level constraint on the representation. It includes the label
object, loss, sampling convention, censoring or missingness convention, and
the evaluation target.

```math
S
=
\left(
\mathcal Y,
\mathcal L,
\mathcal D_{\mathrm{sample}},
\mathcal M_{\mathrm{eval}}
\right).
```

S does not alter the forward map at inference, but it selects which graph
collisions are tolerated during training.

## 3. Permutation equivariance and invariance

Let P be a permutation matrix acting on node order. A graph representation is
well-defined under arbitrary indexing when the construction and context satisfy:

```math
\Gamma_{\mathrm G}
\left(
PH,P_P,\tau_P,\phi_P
\right)
=
P\,
\Gamma_{\mathrm G}
\left(
H,P,\tau,\phi
\right)
P^{\mathsf T},
```

```math
\mathcal C_{\theta}
\left(
PH,PGP^{\mathsf T}
\right)
=
P\,
\mathcal C_{\theta}
\left(
H,G
\right).
```

The readout should then satisfy:

```math
\mathcal R_{\theta}
\left(
P\widetilde H,PGP^{\mathsf T}
\right)
=
\mathcal R_{\theta}
\left(
\widetilde H,G
\right).
```

Consequently:

```math
F_{\theta}
\left(
PH,PGP^{\mathsf T}
\right)
=
F_{\theta}(H,G).
```

This does not mean that graph geometry is discarded. It means that the
representation is invariant to the storage order while remaining sensitive to
the relation object.

A failure of the first two identities is an implementation or representation
problem. A failure of the final identity can be intentional if node order
contains a meaningful sequence, but then the model is no longer ordinary
permutation-invariant graph MIL.

## 4. Context operator matrix

| Context family | Available relation | Local update | Information made available |
| --- | --- | --- | --- |
| coordinate message passing | physical coordinate support | neighbor aggregation and node update | local spatial configurations |
| feature graph attention | encoder-similarity support and edge attribute | type- and edge-conditioned attention | semantic neighbor interactions |
| directed knowledge attention | learned head-tail compatibility | selected-tail message and dual interaction | ordered compatibility triples |
| hierarchical graph context | child graph, parent graph, assignment | fine context, transfer, coarse context | cross-scale summaries |
| generic invariant graph layer | adjacency and optional edge data | shared message-passing map | relation-conditioned node states |

The context operator does not yet determine the slide statistic. It determines
the field that R will summarize.

## 5. Paper-specific forward maps

### 5.1 Patch-GCN

Patch-GCN begins with tissue patches and physical coordinates. Its graph
object is:

```math
G_{\mathrm{PGCN}}
=
\Gamma_{\mathrm{coord}}
\left(
H^{(0)},P
\right),
```

with approximate coordinate k-nearest-neighbor support. Its context stack is
DeepGCN-style local softmax aggregation with residual and dense multi-depth
states:

```math
\widetilde H_{\mathrm{PGCN}}
=
\mathcal C_{\mathrm{DeepGCN}}
\left(
H^{(0)},G_{\mathrm{PGCN}}
\right).
```

The graph-level readout is global scalar attention:

```math
z_{\mathrm{PGCN}}
=
\sum_{v=1}^{N}
\alpha_v\widetilde h_v,
\qquad
\alpha_v
=
\frac{\exp(g(\widetilde h_v))}
{\sum_u\exp(g(\widetilde h_u))}.
```

The surviving statistic is a weighted first moment of spatially
contextualized, multiscale node states. In a survival application, the head
maps this vector to a patient-level risk score.

```math
\boxed{
G=\text{coordinate support},
\quad
C=\text{DeepGCN-style local context},
\quad
R=\text{global attention},
\quad
S=\text{patient-level survival supervision}
}
```

### 5.2 HEAT

HEAT constructs a heterogeneous graph from KimiaNet patch features, feature
space k-nearest-neighbor support, teacher-generated node types, and Pearson
edge attributes:

```math
G_{\mathrm{HEAT}}
=
\left(
V,E,\mathcal A,\mathcal E,X,\tau,\phi
\right).
```

Its context uses type-specific query, key, and value projections together with
an edge-attribute projection:

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

The node update aggregates incoming edge-conditioned values. PL-Pool then
creates a row for each fixed semantic type:

```math
S_a
=
\mathcal R_a
\left(
\{\widetilde h_v:\tau(v)=a\}
\right).
```

A graph-level readout acts on the type matrix:

```math
z_{\mathrm{HEAT}}
=
\mathcal R_{\mathrm{graph}}
\left(
S_{a_1},\ldots,S_{a_{|\mathcal A|}}
\right).
```

```math
\boxed{
G=\text{feature graph plus types and edge attributes},
\quad
C=\text{heterogeneous edge-conditioned attention},
\quad
R=\text{fixed type rows plus graph readout},
\quad
S=\text{slide-level task supervision with teacher-generated attributes}
}
```

### 5.3 WiKG

WiKG creates head and tail states:

```math
h_v=W_hx_v,
\qquad
t_u=W_tx_u.
```

It selects directed support from head-tail compatibility:

```math
\mathcal N_K(v)
=
\mathrm{TopK}_{u\ne v}
\left(
h_v^{\mathsf T}t_u
\right).
```

The selected relation state is:

```math
r_{vu}
=
\widetilde\omega_{vu}t_u
+
\left(
1-\widetilde\omega_{vu}
\right)h_v.
```

Knowledge-aware attention produces a selected-neighbor message:

```math
h_{\mathcal N(v)}
=
\sum_{u\in\mathcal N_K(v)}
\pi_{vu}t_u.
```

The dual interaction update is:

```math
h_v^{+}
=
\sigma_1
\left(
W_1(h_v+h_{\mathcal N(v)})
\right)
+
\sigma_2
\left(
W_2(h_v\odot h_{\mathcal N(v)})
\right).
```

The paper-level graph readout is mean or coordinatewise max:

```math
z_{\mathrm{WiKG}}^{\mathrm{mean}}
=
\frac{1}{N}\sum_vh_v^{+},
```

```math
\left[
z_{\mathrm{WiKG}}^{\mathrm{max}}
\right]_r
=
\max_v[h_v^{+}]_r.
```

```math
\boxed{
G=\text{directed learned top-k support},
\quad
C=\text{knowledge-aware directed interaction},
\quad
R=\text{mean or coordinatewise max},
\quad
S=\text{slide-level classification supervision}
}
```

### 5.4 HACT

HACT is a hierarchical object, not a flat graph with a cosmetic second scale:

```math
G_{\mathrm{HACT}}
=
\left(
G_{\mathrm C},
G_{\mathrm T},
B_{\mathrm C\to T}
\right).
```

The assignment matrix has shape:

```math
B_{\mathrm C\to T}
\in
\{0,1\}^{N_{\mathrm C}\times N_{\mathrm T}},
\qquad
\sum_{b=1}^{N_{\mathrm T}}B_{ab}=1.
```

Cell context is followed by assignment transfer:

```math
U_{\mathrm C\to T}
=
B_{\mathrm C\to T}^{\mathsf T}
\widetilde H_{\mathrm C}.
```

Tissue context then acts on tissue features and transferred cell summaries:

```math
\widetilde H_{\mathrm T}
=
\mathcal C_{\mathrm T}
\left(
\left[
H_{\mathrm T}
\middle\Vert
U_{\mathrm C\to T}
\right],
G_{\mathrm T}
\right).
```

A coarse graph readout produces the slide representation:

```math
z_{\mathrm{HACT}}
=
\mathcal R_{\mathrm T}
\left(
\widetilde H_{\mathrm T}
\right).
```

```math
\boxed{
G=\text{cell graph, tissue graph, and assignment},
\quad
C=\text{cell context, transfer, tissue context},
\quad
R=\text{coarse tissue readout},
\quad
S=\text{region-level classification supervision in the source task}
}
```

HACT was originally proposed for annotated tissue regions and breast-cancer
subtyping. Applying the operator to generic WSI survival is an extension and
should be named as such.

## 6. Readout operator matrix

| Readout | Formula | Surviving statistic | Duplication response | Main collision |
| --- | --- | --- | --- | --- |
| sum | sum of contextualized nodes | total first moment | doubles under identical duplication | count and burden configurations with equal total |
| mean | normalized sum | empirical first moment | invariant under identical duplication | distributions with equal mean |
| max | coordinatewise maximum | coordinatewise upper tail | invariant under duplication | all submaximal and multiplicity structure |
| attention | normalized score-weighted sum | learned weighted first moment | denominator changes under duplication | different score/value mixtures with equal vector |
| typed | one readout per semantic type | type-conditioned statistics | type counts can matter | within-type arrangement |
| hierarchical | transfer then coarse readout | coarse multiscale statistic | depends on assignment and local operator | fine fields in assignment nullspace |

The readout matrix is incomplete without specifying whether the node states are
raw or contextualized:

```math
\mathcal R_{\mathrm{raw}}(H^{(0)})
\ne
\mathcal R_{\mathrm{context}}
\left(
\mathcal C(H^{(0)},G)
\right)
```

in general. Calling both “mean pooling” hides the distinction.

## 7. Exact readout properties

### Sum

```math
z^{\mathrm{sum}}
=
\sum_{v=1}^{N}\widetilde h_v.
```

Equal duplication gives:

```math
z^{\mathrm{sum}}(H\uplus H)
=
2z^{\mathrm{sum}}(H).
```

The sum preserves total mass but makes count and representation scale
inseparable unless count is explicitly modeled.

### Mean

```math
z^{\mathrm{mean}}
=
\frac{1}{N}
\sum_{v=1}^{N}\widetilde h_v.
```

It is the first moment of the empirical measure:

```math
\widehat\mu
=
\frac{1}{N}
\sum_{v=1}^{N}\delta_{\widetilde h_v},
\qquad
z^{\mathrm{mean}}
=
\int x\,d\widehat\mu(x).
```

It is blind to count under identical duplication but not necessarily blind to
count through C, because degree normalization and graph construction can change
the contextualized states.

### Max

```math
[z^{\mathrm{max}}]_r
=
\max_v[\widetilde h_v]_r.
```

Different coordinates can be supplied by different nodes:

```math
v_r^\star
\in
\arg\max_v[\widetilde h_v]_r.
```

The max vector is therefore not necessarily the state of an actual node.

### Attention

```math
z^{\mathrm{attn}}
=
\sum_v\alpha_v\widetilde h_v,
\qquad
\alpha_v
=
\frac{\exp(s_v)}{\sum_u\exp(s_u)}.
```

It lies in the convex hull of contextualized states:

```math
z^{\mathrm{attn}}
\in
\mathrm{conv}
\{\widetilde h_v\}_{v=1}^{N}.
```

The coefficient is not a probability that a node is positive. It is a
normalization coefficient in a representation operator.

### Typed rows

```math
Z_{a,:}
=
\mathcal R_a
\left(
\{\widetilde h_v:\tau(v)=a\}
\right).
```

A fixed row coordinate preserves the semantic identity of type a across
slides. It does not preserve within-type spatial order, pairwise geometry, or
teacher uncertainty unless those are added to R.

### Hierarchical transfer

```math
U_{\mathrm C\to T}
=
B^{\mathsf T}\widetilde H_{\mathrm C}.
```

If B-transpose Delta is zero, the transfer path cannot distinguish the two cell
fields:

```math
B^{\mathsf T}\Delta=0
\Longrightarrow
B^{\mathsf T}
\left(
\widetilde H_{\mathrm C}+\Delta
\right)
=
B^{\mathsf T}\widetilde H_{\mathrm C}.
```

## 8. Context and readout do not commute

A graph context map and a readout generally do not commute. Let M be a
nodewise linear map and let C be a relation-dependent message-passing
operator. In general:

```math
\mathcal R
\left(
\mathcal C(H,G)
\right)
\ne
\mathcal C_{\mathrm{slide}}
\left(
\mathcal R(H),G
\right).
```

The right-hand side is often not even well-defined, because a graph context
needs node-level relations that R has already removed.

For a two-node scalar example:

```math
H
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\widetilde A
=
\frac{1}{2}
\begin{bmatrix}
1&1\\
1&1
\end{bmatrix}.
```

Context then mean gives:

```math
\frac{1}{2}
\mathbf 1^{\mathsf T}
\widetilde A H
=
\frac{1}{2}.
```

Mean then context cannot reconstruct the two-node relation because the scalar
mean is all that remains:

```math
\mathcal R_{\mathrm{mean}}(H)
=
\frac{1}{2}.
```

The distinction is the mathematical reason that “graph layer plus mean” is not
equivalent to “mean plus graph layer.”

## 9. Same C, different R

Fix a contextual field:

```math
\widetilde H
=
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix}.
```

Mean, max, and sum produce:

```math
z^{\mathrm{mean}}=\frac{1}{3},
\qquad
z^{\mathrm{max}}=1,
\qquad
z^{\mathrm{sum}}=1.
```

The graph context is identical. The surviving statistic differs. For a sparse
positive-instance target, max can preserve a witness that mean attenuates. For
a burden target, mean can preserve a fraction that max discards.

This is why an ablation that changes both the graph layer and pooling cannot
attribute a performance change to C or R.

## 10. Same R, different G

Fix mean readout and use the two supports from the node counterexample:

```math
z_1
=
\frac{1}{2}
\mathbf 1^{\mathsf T}
\widetilde A_1H,
\qquad
z_2
=
\frac{1}{2}
\mathbf 1^{\mathsf T}
\widetilde A_2H.
```

For the chosen H, they collide even though the node fields differ. With a max
readout they separate. Thus changing G can be visible only through a particular
R.

A graph ablation must therefore report both:

```math
\Delta_{\mathrm{node}}
=
d
\left(
\mathcal C(H,G),
\mathcal C(H,G')
\right),
```

```math
\Delta_{\mathrm{slide}}
=
d
\left(
\mathcal R(\mathcal C(H,G)),
\mathcal R(\mathcal C(H,G'))
\right).
```

A small slide-level difference does not prove that the graph context is unused;
R may have erased the difference.

## 11. Geometry matrix

| Geometry object | Relation meaning | What it preserves | What it can invent |
| --- | --- | --- | --- |
| no explicit geometry | only feature multiset | permutation-invariant evidence | no spatial relation |
| coordinates | physical proximity | local layout and scale-dependent neighborhoods | false proximity across boundaries |
| coordinate grid | regular local lattice | translational or window structure | artifacts from missing tissue |
| feature kNN | encoder similarity | semantic similarity under the encoder | long-range texture shortcuts |
| learned directed top-k | task-conditioned compatibility | ordered support and relation roles | unstable or label-shortcut edges |
| hierarchy | fine-to-coarse membership | parent-child scale structure | segmentation and assignment bias |
| heterogeneous graph | typed nodes and edge attributes | semantic relation categories | teacher and attribute noise |

Geometry is an input hypothesis. It is not a guarantee that the encoded relation
is causal, anatomical, or stable across cohorts.

## 12. Supervision matrix

The graph representation is trained through a task-specific loss. Examples:

| Task | Head output | Typical supervision | What can be rewarded |
| --- | --- | --- | --- |
| classification | class logits | cross-entropy | any graph statistic correlated with class |
| survival risk | scalar risk | Cox partial likelihood or ranking loss | patient ordering, not necessarily calibrated survival |
| discrete hazard | horizon logits | masked hazard likelihood | horizon-specific event probabilities |
| retrieval | embedding or similarity | contrastive or ranking loss | neighborhood geometry in the training cohort |
| multimodal alignment | graph or slide embedding | image-text alignment | cross-modal shared directions |

For a classification head:

```math
\mathcal L_{\mathrm{cls}}
=
-\sum_{c=1}^{C}
y_c
\log
\left[
\mathrm{softmax}(Wz+b)
\right]_c.
```

For a Cox risk score:

```math
\eta_i=w^{\mathsf T}z_i.
```

The partial log-likelihood ranks risk scores within risk sets:

```math
\ell_{\mathrm{Cox}}
=
\sum_{i:\delta_i=1}
\left[
\eta_i
-
\log
\sum_{j\in\mathcal R(T_i)}
\exp(\eta_j)
\right].
```

A graph statistic can be useful for ordering while being inadequate for
calibration. The task loss decides which representation collisions are
acceptable.

## 13. Complexity matrix

Let N be the node count, E the edge count, d the hidden width, K the selected
degree, and A the number of semantic types.

| Operator | Typical cost | Main scaling risk |
| --- | --- | --- |
| sparse message passing | order E d-squared | dense or high-degree support |
| coordinate kNN construction | implementation-dependent, often superlinear | large WSI node count |
| feature kNN construction | implementation-dependent, often superlinear | memory and approximate-index shift |
| dense WiKG candidate scores | order N-squared d | all head-tail pairs before top-k |
| selected WiKG context | order N K d-squared | K and relation projections |
| global sum or mean | order N d | count or density statistic |
| global attention readout | order N d plus score network | score network and normalization |
| HEAT typed rows | order N d plus type/edge projections | many types and rare rows |
| HACT assignment transfer | order nnz(B) d | fine-cell count and assignment storage |

A complexity claim should state whether graph construction is included. A model
with linear message passing can still have a quadratic feature-graph
construction stage.

## 14. Information budget

Before readout, the contextualized field has order N d real-valued coordinates:

```math
\widetilde H
\in
\mathbb R^{N\times d}.
```

A slide vector has q coordinates:

```math
z
\in
\mathbb R^q.
```

When q is fixed and N grows, R is a severe compression. The effective
information budget can be summarized by the rank of a local Jacobian:

```math
J_{\mathrm R}
=
\frac{\partial z}{\partial\mathrm{vec}(\widetilde H)}.
```

For sum and mean, the Jacobian has repeated blocks:

```math
\frac{\partial z^{\mathrm{mean}}}
{\partial\widetilde h_v}
=
\frac{1}{N}I_d.
```

For attention, the derivative includes both value and score paths:

```math
\frac{\partial z^{\mathrm{attn}}}{\partial\widetilde h_v}
=
\alpha_v I_d
+
\sum_u
\widetilde h_u
\frac{\partial\alpha_u}{\partial\widetilde h_v}.
```

The score path is why attention readout can change all coefficients when one node
changes. It is also why a high attention weight does not equal an isolated
causal contribution.

For HACT, the transfer Jacobian is:

```math
\frac{\partial U_{\mathrm C\to T}}
{\partial\mathrm{vec}(\widetilde H_{\mathrm C})}
=
B^{\mathsf T}\otimes I_d.
```

Its nullspace dimension is at least:

```math
d
\left(
N_{\mathrm C}
-
\mathrm{rank}(B)
\right).
```

The assignment transfer therefore has an explicit representation budget before
the tissue graph is even applied.

## 15. Equivalence and non-equivalence tests

### 15.1 Same patch set, different topology

```math
\{x_v\}_{v\in V}
=
\{x_v'\}_{v\in V'},
\qquad
G\ne G'.
```

A graph model should differ only if C or R uses G:

```math
F(H,G)
\ne
F(H',G')
```

for some task-relevant pair. A set model must collide if the features are
identical as a multiset.

### 15.2 Same graph, different edge attributes

```math
G=(V,E,X,\phi),
\qquad
G'=(V,E,X,\phi').
```

HEAT can distinguish these because phi enters C. A homogeneous graph layer that
ignores phi cannot.

### 15.3 Same type rows, different within-type geometry

```math
\mathcal R_a(\widetilde H_a)
=
\mathcal R_a(\widetilde H_a')
\qquad
\forall a.
```

A downstream row readout must collide unless spatial information was retained
inside the row or passed through another graph path.

### 15.4 Same assignment transfer, different cells

```math
B^{\mathsf T}\widetilde H_{\mathrm C}
=
B^{\mathsf T}\widetilde H_{\mathrm C}'.
```

A tissue-only HACT path cannot distinguish the two cell fields.

## 16. Design gaps exposed by the matrix

The matrix identifies open design axes without declaring that every empty cell is
a good idea.

### 16.1 Geometry-preserving distribution readout

Graph context can preserve local geometry, while a mean readout discards most of
the contextual distribution. A richer design could use per-region moments:

```math
z_r
=
\left[
\frac{1}{|V_r|}
\sum_{v\in V_r}\widetilde h_v,
\;
\mathrm{vec}
\left(
\widehat{\mathrm{Cov}}_r
\right)
\right].
```

This increases the statistic sent to the head but still requires a region
definition and a stability analysis.

### 16.2 Uncertainty-aware typed context

Instead of a hard type tau-v, use a probability vector pi-v:

```math
\pi_v\in\Delta^{|\mathcal A|-1}.
```

A type-conditioned transfer can be softened:

```math
z_a
=
\sum_v\pi_{va}\widetilde h_v.
```

This preserves teacher uncertainty but changes the semantic row from a hard
partition to an expected assignment.

### 16.3 Stable learned support

A differentiable soft support can replace a hard top-k support:

```math
w_{vu}
=
\frac{\exp(s_{vu}/\tau)}
{\sum_r\exp(s_{vr}/\tau)}.
```

As tau tends to zero, the weights become concentrated but support instability
increases. A stability-regularized design could penalize changes under feature
perturbation:

```math
\mathcal L_{\mathrm{stab}}
=
\sum_{(v,u)}
\left|
w_{vu}(H)
-
w_{vu}(H+\epsilon)
\right|.
```

This trades sharp selection against robustness.

### 16.4 Hierarchy with within-region dispersion

A sum transfer preserves total cell mass but loses dispersion. Add second
moments inside each parent:

```math
m_b
=
\frac{1}{n_b}
\sum_{a:B_{ab}=1}
\widetilde h_a,
```

```math
V_b
=
\frac{1}{n_b}
\sum_{a:B_{ab}=1}
\left(
\widetilde h_a-m_b
\right)
\left(
\widetilde h_a-m_b
\right)^{\mathsf T}.
```

This removes some assignment-nullspace collisions at a higher computational and
estimation cost.

### 16.5 Survival-specific graph readout

A scalar risk vector can be replaced by horizon-conditioned graph readouts:

```math
z_i^{(k)}
=
\mathcal R_{\theta,k}
\left(
\widetilde H_i,G_i
\right),
\qquad
\eta_i^{(k)}
=
w_k^{\mathsf T}z_i^{(k)}.
```

This lets different regions matter at different horizons, but it increases the
risk of horizon-specific attention instability and multiple-testing artifacts.

## 17. Reporting template

A graph MIL method is minimally specified when the paper reports:

```text
1. node definition, patch scale, and patch encoder
2. graph construction inputs and support rule
3. directionality, self-loops, edge weights, and normalization
4. node types and edge attributes, including teacher provenance
5. context update with tensor shapes
6. depth, receptive field, and any residual or dense connections
7. graph readout formula and whether states are contextualized
8. count, missing-type, and empty-graph conventions
9. task head and exact loss
10. graph-specific ablations and intervention tests
11. graph-construction cost and inference memory
12. patient-level split and any cross-slide index construction
```

The C/R/G/S form is:

```math
\begin{aligned}
G &: \Gamma_{\mathrm G}(\text{features, coordinates, attributes}),
\\
C &: \mathcal C_{\theta}(\text{node field},G),
\\
R &: \mathcal R_{\theta}(\text{contextualized field},G),
\\
S &: \left(\text{labels},\text{loss},\text{sampling},\text{evaluation}\right).
\end{aligned}
```

A paper that reports only the GNN layer has not fully specified the slide
representation.

## 18. Final paper matrix

| Method | G: relation object | C: context operator | R: surviving statistic | S: task constraint |
| --- | --- | --- | --- | --- |
| Patch-GCN | coordinate kNN patch graph | local softmax graph context with multiscale states | global weighted first moment | patient-level survival or slide task |
| HEAT | feature kNN with node types and Pearson edge attributes | heterogeneous edge-conditioned attention | fixed type-conditioned rows | slide-level task with teacher-generated attributes |
| WiKG | learned directed head-tail top-k graph | knowledge-aware local attention and dual interaction | first moment or coordinatewise extrema | slide classification |
| HACT | cell graph, tissue graph, hard assignment | fine context, assignment transfer, coarse context | hierarchical tissue summary | region-level classification in source paper; broader tasks are extensions |
| generic graph MIL | explicit support and edge object | relation-conditioned message passing | chosen invariant or typed readout | task loss selects useful collisions |

## 19. Bottom line

The unified graph MIL map is:

```math
\boxed{
F_{\theta}
=
\mathcal H_{\theta}
\circ
\mathcal R_{\theta}
\circ
\mathcal C_{\theta}
\circ
\Gamma_{\mathrm G}
}
```

The important comparison is not merely:

```text
Patch-GCN versus HEAT versus WiKG versus HACT
```

It is:

```math
\boxed{
\begin{aligned}
\text{G}:&\ \text{what relation is treated as available?}
\\
\text{C}:&\ \text{how does that relation alter node states?}
\\
\text{R}:&\ \text{what statistic survives the slide bottleneck?}
\\
\text{S}:&\ \text{which collisions does the task reward or tolerate?}
\end{aligned}
}
```

A graph representation is mathematically legible only when all four answers are
explicit.
