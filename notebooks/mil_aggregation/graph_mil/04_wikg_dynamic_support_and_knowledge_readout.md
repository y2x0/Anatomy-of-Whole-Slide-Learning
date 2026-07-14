# WiKG: Dynamic Directed Support, Knowledge-Aware Attention, and Graph Readout

Source:

Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.

[arXiv](https://arxiv.org/abs/2403.07719)

[CVPR open-access paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Dynamic_Graph_Representation_with_Knowledge-aware_Attention_for_Histopathology_Whole_Slide_CVPR_2024_paper.pdf)

WiKG represents a WSI as a dynamic directed graph. It does not define support
from physical coordinates. Instead, it gives every patch two roles:

```math
\text{head embedding}
\qquad
\text{tail embedding}.
```

The head role asks which other patches should contribute to a patch. The tail
role describes what a patch contributes when it is selected as a neighbor. The
resulting graph contains directed edges and high-dimensional edge embeddings.

The paper's forward map is:

```math
\text{patch features}
\longrightarrow
\text{head/tail projections}
\longrightarrow
\text{directed top-k support and edge embeddings}
\longrightarrow
\text{knowledge-aware attention}
\longrightarrow
\text{mean or max graph readout}
\longrightarrow
\text{classification}.
```

The central inductive bias is:

```math
\boxed{
\text{patch interactions should be learned from directed head-tail compatibility,
not fixed by physical proximity}
}
```

## 1. WSI as a dynamic graph

Let a WSI contain N retained patch instances:

```math
X
=
\{x_i\}_{i=1}^{N}.
```

A feature encoder produces:

```math
f_i
=
f(x_i)
\in
\mathbb R^{d_0}.
```

Stack the features:

```math
F
=
\begin{bmatrix}
f_1^{\mathsf T}\\
\vdots\\
f_N^{\mathsf T}
\end{bmatrix}
\in
\mathbb R^{N\times d_0}.
```

WiKG projects each patch into a head and a tail embedding:

```math
h_i
=
W_h f_i,
\qquad
t_i
=
W_t f_i,
```

where:

```math
h_i,t_i
\in
\mathbb R^d.
```

The dynamic graph object is:

```math
G
=
(V,E,F_{\mathrm{HT}},R),
```

where:

| Symbol | Meaning |
| --- | --- |
| V | patch nodes |
| E | directed top-k support |
| F_HT | head and tail embedding collection |
| R | directed edge-embedding collection |

For each selected directed edge from tail j into head i, the graph contains a
triplet:

```math
(h_i,r_{ij},t_j).
```

The edge orientation is part of the object:

```math
j\longrightarrow i
\quad
\text{does not imply}
\quad
i\longrightarrow j.
```

## 2. Head and tail projections

The head projection is:

```math
H
=
FW_h^{\mathsf T},
```

and the tail projection is:

```math
T
=
FW_t^{\mathsf T}.
```

For node i and candidate node j:

```math
h_i
=
W_hf_i,
\qquad
t_j
=
W_tf_j.
```

The two matrices need not be equal:

```math
W_h
\ne
W_t
\quad
\text{in general}.
```

This asymmetry allows a patch to have different roles as a receiver and as a
message source. In a symmetric similarity graph, the same representation often
plays both roles. WiKG makes the distinction explicit.

The head-tail score is:

```math
\ell_{ij}
=
h_i^{\mathsf T}t_j.
```

A scaled implementation may use:

```math
\ell_{ij}
=
\frac{
h_i^{\mathsf T}t_j
}{
\sqrt d
}.
```

The scale changes numerical conditioning but not the conceptual direction of
the relation.

## 3. First relation score

The paper displays a normalized head-tail similarity:

```math
\omega_{ij}
=
\frac{
h_i^{\mathsf T}t_j
}{
\sum_{r=1}^{N}
h_i^{\mathsf T}t_r
}.
```

This equation should be read as the paper's displayed relation score. A
denominator based on raw signed dot products is not a probability normalization
when the denominator can be zero or negative.

The implementation-level construction exposes the distinction more clearly.
It computes attention logits:

```math
\ell_{ij}
=
\left(
h_i\cdot\mathrm{scale}
\right)^{\mathsf T}
t_j,
```

then selects top-k logits and applies a softmax to the selected values:

```math
\widetilde\omega_{ij}
=
\frac{
\exp(\ell_{ij})
}{
\sum_{r\in\mathcal N_K(i)}
\exp(\ell_{ir})
},
\qquad
j\in\mathcal N_K(i).
```

The faithful conceptual statement is:

```text
head-tail compatibility determines candidate neighbors;
the selected compatibility values determine edge interpolation weights.
```

Do not silently replace the paper's displayed raw normalization with a
global class probability. This is a topology score, not a patch posterior.

## 4. Dynamic top-k support

For each head node i, select k candidate tails:

```math
\mathcal N_K(i)
=
\mathrm{TopK}_{j}
\left(
\ell_{ij}
\right).
```

Equivalently:

```math
j\in\mathcal N_K(i)
\Longleftrightarrow
\ell_{ij}
\text{ is among the k largest scores for head }i.
```

The directed adjacency is:

```math
A_{ij}
=
\mathbf 1
\left\{
j\in\mathcal N_K(i)
\right\}.
```

The out-neighborhood of a head is the set of tails that can send information
into it. Depending on matrix convention, this may be stored as row i with
column j, but the semantic direction remains:

```math
t_j
\longrightarrow
h_i.
```

The graph is dynamic because:

```math
A
=
A(F;W_h,W_t).
```

Changing the node features or projection parameters can change the support.

The support is not physical-coordinate kNN:

```math
A_{\mathrm{WiKG}}
=
\Gamma_{\mathrm{head\text{-}tail}}(F),
\qquad
A_{\mathrm{coord}}
=
\Gamma_{\mathrm{coord}}(C).
```

WiKG uses the first relation.

## 5. Directedness

The score from j to i is:

```math
\ell_{ij}
=
h_i^{\mathsf T}t_j.
```

The reverse score is:

```math
\ell_{ji}
=
h_j^{\mathsf T}t_i.
```

Because head and tail projections differ:

```math
h_i^{\mathsf T}t_j
\ne
h_j^{\mathsf T}t_i
\quad
\text{in general}.
```

Therefore:

```math
j\in\mathcal N_K(i)
\centernot\Longrightarrow
i\in\mathcal N_K(j).
```

A directed relation is not the same as an undirected edge with an asymmetric
weight. The source and target roles determine different projections and
different candidate sets.

## 6. Edge embedding

For each selected tail j of head i, WiKG forms a directed edge embedding:

```math
r_{ij}
=
\omega_{ij}t_j
+
\left(
1-\omega_{ij}
\right)h_i.
```

Using the implementation's selected softmax weight gives:

```math
r_{ij}
=
\widetilde\omega_{ij}t_j
+
\left(
1-\widetilde\omega_{ij}
\right)h_i.
```

When the weight lies between zero and one, the edge lies on the line segment
between the tail and head embeddings:

```math
r_{ij}
\in
\mathrm{conv}
\{h_i,t_j\}.
```

The edge is not just a scalar. It has the same feature dimension as h and t.
It encodes a relation-conditioned blend:

```math
r_{ij}
=
h_i
+
\omega_{ij}
\left(
t_j-h_i
\right).
```

Thus:

```math
r_{ij}-h_i
=
\omega_{ij}
\left(
t_j-h_i
\right).
```

If tail and head are close, the edge representation has limited geometric
span. If they differ strongly, the relation contains a larger direction from
the head toward the tail.

## 7. Triplet representation

The selected graph can be written as a set of directed triples:

```math
\mathcal E_{\mathrm{triplet}}
=
\left\{
\left(
h_i,r_{ij},t_j
\right):
j\in\mathcal N_K(i)
\right\}.
```

The graph representation is:

```math
G
=
\left(
V,
E,
F,
R
\right),
```

where F contains head and tail embeddings and R contains edge embeddings.

The three terms play distinct roles:

```text
head:
    receiver query and target-specific context

tail:
    message content sent into the head

edge:
    relation-conditioned state that changes knowledge-aware compatibility
```

This is the reason the method is described as knowledge-graph-like. The patch
nodes are not biological entities with an externally supplied relation
ontology; the relation embedding is learned from head-tail compatibility.

## 8. Knowledge-aware neighbor score

For head i and selected tail j, define the unnormalized knowledge-aware score:

```math
u_{ij}
=
t_j^{\mathsf T}
\tanh
\left(
h_i+r_{ij}
\right).
```

The elementwise hyperbolic tangent is applied to the head-plus-edge relation
state. The score is signed:

```math
u_{ij}
\in
\mathbb R.
```

The tail vector then measures compatibility with the relation-conditioned head
state.

The normalized knowledge-aware attention is:

```math
\pi_{ij}
=
\frac{
\exp(u_{ij})
}{
\sum_{r\in\mathcal N_K(i)}
\exp(u_{ir})
}.
```

For every head i:

```math
\sum_{j\in\mathcal N_K(i)}
\pi_{ij}
=
1.
```

This is the second normalization in WiKG. It should be distinguished from the
first score used to construct support and edge embeddings.

## 9. Neighbor representation

The first-order neighbor representation for head i is:

```math
h_{\mathcal N(i)}
=
\sum_{j\in\mathcal N_K(i)}
\pi_{ij}t_j.
```

It is a convex combination of selected tail embeddings:

```math
h_{\mathcal N(i)}
\in
\mathrm{conv}
\{t_j:j\in\mathcal N_K(i)\}.
```

The relation embedding affects the weights through u-ij, but the value being
aggregated is t-j. The edge is used to decide how much of each tail
contributes, not necessarily concatenated as the message value.

The neighbor statistic is a directed, knowledge-aware weighted first moment:

```math
h_{\mathcal N(i)}
=
\mathbb E_{\pi_i}
\left[
t_j
\right].
```

This is local to the learned support, not a global WSI pooling operation.

## 10. Dual interaction update

WiKG combines the original head with the neighbor summary through two channels:

```math
h_i^{+}
=
\sigma_1
\left(
W_1
\left(
h_i+h_{\mathcal N(i)}
\right)
\right)
+
\sigma_2
\left(
W_2
\left(
h_i\odot h_{\mathcal N(i)}
\right)
\right).
```

The first channel is additive:

```math
a_i
=
\sigma_1
\left(
W_1
\left(
h_i+h_{\mathcal N(i)}
\right)
\right).
```

The second is coordinatewise multiplicative:

```math
b_i
=
\sigma_2
\left(
W_2
\left(
h_i\odot h_{\mathcal N(i)}
\right)
\right).
```

The coordinate interaction is:

```math
\left[
h_i\odot h_{\mathcal N(i)}
\right]_r
=
[h_i]_r
[h_{\mathcal N(i)}]_r.
```

The output is:

```math
h_i^{+}
=
a_i+b_i.
```

The product is a learned feature interaction. It is not automatically a
biological interaction between two cell types, because both vectors are
already learned projections of patch-level features and aggregated neighbors.

## 11. Why the edge enters before the dual interaction

The path from a tail to the updated head is:

```math
t_j
\longrightarrow
r_{ij}
\longrightarrow
u_{ij}
\longrightarrow
\pi_{ij}
\longrightarrow
h_{\mathcal N(i)}
\longrightarrow
h_i^{+}.
```

The edge affects both support-local relation selection and the final message
weight. The head receives a message whose content is not simply the mean of
selected tails.

If the edge is removed from the knowledge score, the neighbor weight becomes a
head-tail score:

```math
u_{ij}^{\mathrm{no\ edge}}
=
t_j^{\mathsf T}\tanh(h_i).
```

This is a different context operator. The edge-aware model can distinguish
two tails with similar t-j if their relation embeddings differ.

## 12. Two tails with the same tail vector

Suppose:

```math
t_a=t_b,
```

but:

```math
r_{ia}\ne r_{ib}.
```

Then their edge-conditioned scores can differ:

```math
u_{ia}
=
t_a^{\mathsf T}\tanh(h_i+r_{ia}),
```

```math
u_{ib}
=
t_b^{\mathsf T}\tanh(h_i+r_{ib}).
```

Thus:

```math
u_{ia}\ne u_{ib}
\quad
\Longrightarrow
\quad
\pi_{ia}\ne\pi_{ib}
\quad
\text{in general}.
```

The relation path can distinguish equal tail embeddings. A plain mean or GAT
that uses only tail features cannot recover that distinction.

## 13. Two heads with the same head vector

Suppose:

```math
h_i=h_r,
```

but their candidate tail sets differ:

```math
\mathcal N_K(i)
\ne
\mathcal N_K(r).
```

The neighbor summaries can differ:

```math
h_{\mathcal N(i)}
\ne
h_{\mathcal N(r)}.
```

Even equal head features do not imply equal updated states because the learned
directed support is part of the input.

## 14. Hard top-k support is a discontinuous operator

Let the candidate scores for head i be ordered:

```math
\ell_{i,(1)}
\ge
\ell_{i,(2)}
\ge
\cdots
\ge
\ell_{i,(N)}.
```

The support contains the first k indices. If the kth and k-plus-one-th scores
are close:

```math
\ell_{i,(k)}
\approx
\ell_{i,(k+1)},
```

a small perturbation can swap membership:

```math
\ell_{i,(k)}
>
\ell_{i,(k+1)}
\longrightarrow
j_{k}\in\mathcal N_K(i),
```

```math
\ell_{i,(k)}
<
\ell_{i,(k+1)}
\longrightarrow
j_{k+1}\in\mathcal N_K(i).
```

The support map is piecewise constant in the scores and discontinuous at ties.
The downstream edge set, relation embeddings, and attention distribution can
all change at that boundary.

The top-k value gradients do not describe the discrete effect of changing
membership. A graph-aware deletion or support perturbation is needed to test
that effect.

## 15. Softmax on selected support

After top-k selection, the knowledge-aware weights are:

```math
\pi_{ij}
=
\frac{\exp(u_{ij})}
{\sum_{r\in\mathcal N_K(i)}\exp(u_{ir})}.
```

Their Jacobian is:

```math
\frac{
\partial\pi_{ij}
}{
\partial u_{ir}
}
=
\pi_{ij}
\left(
\mathbf 1\{j=r\}
-
\pi_{ir}
\right).
```

Increasing the score of one selected tail decreases the normalized mass of
other selected tails. The relation-aware message is competitive:

```math
\frac{
\partial h_{\mathcal N(i)}
}{
\partial u_{ir}
}
=
\pi_{ir}
\left(
t_r-h_{\mathcal N(i)}
\right).
```

This has the same centered-value structure as ordinary softmax attention, but
the score u-ir includes the edge state.

## 16. Graph readout

After the dynamic graph update, WiKG applies a global readout to updated head
features:

```math
H^{+}
=
\{h_i^{+}\}_{i=1}^{N}.
```

The paper describes mean or max readout.

Mean readout:

```math
z^{\mathrm{mean}}
=
\frac{1}{N}
\sum_{i=1}^{N}
h_i^{+}.
```

Coordinatewise max readout:

```math
\left[
z^{\mathrm{max}}
\right]_r
=
\max_{1\le i\le N}
[h_i^{+}]_r.
```

The classifier is:

```math
\widehat Y
=
\mathrm{softmax}
\left(
W_{\mathrm{cls}}z+b_{\mathrm{cls}}
\right).
```

The released training configuration may use mean pooling. The paper-level
architecture permits mean or max as the global readout; these should not be
reported as the same surviving statistic.

## 17. Cross-entropy objective

For M training slides, C classes, and one-hot targets Y-m-c:

```math
\mathcal L_{\mathrm{CE}}
=
-
\frac{1}{M}
\sum_{m=1}^{M}
\sum_{c=1}^{C}
Y_{m,c}
\log
\left(
\widehat Y_{m,c}
\right).
```

The gradient reaches:

```math
\text{classifier}
\longrightarrow
\text{global readout}
\longrightarrow
\text{updated heads}
\longrightarrow
\pi
\longrightarrow
r
\longrightarrow
\text{top-k relation scores}
\longrightarrow
W_h,W_t.
```

The hard top-k index is not ordinarily differentiable through membership. The
selected logits and edge constructions receive gradient conditional on the
current support; support changes occur through the discrete top-k operation.

## 18. Mean versus max surviving statistics

Mean retains a first moment:

```math
z^{\mathrm{mean}}
=
\mathbb E_{\widehat\mu_H}
\left[
h_i^{+}
\right].
```

Max retains coordinatewise upper extremes:

```math
z_r^{\mathrm{max}}
=
\sup_i[h_i^{+}]_r.
```

Two updated node fields can have the same mean but different rare extremes:

```math
\frac{1}{N}
\sum_i h_i^{+}
=
\frac{1}{N'}
\sum_i h_i^{+\prime},
```

while:

```math
\max_i h_{i,r}^{+}
\ne
\max_i h_{i,r}^{+\prime}.
```

The graph context is shared, but the graph-level task information that survives
is readout-dependent.

## 19. Mean injection and implementation boundary

Some released implementations may inject a slide-level mean into each patch
before constructing dynamic relations. If used, define:

```math
\overline f
=
\frac{1}{N}
\sum_{i=1}^{N}f_i,
```

```math
\widetilde f_i
=
\frac{1}{2}
\left(
f_i+\overline f
\right).
```

The head and tail projections then become:

```math
h_i
=
W_h\widetilde f_i,
\qquad
t_i
=
W_t\widetilde f_i.
```

This creates a global first-moment path before top-k support selection:

```math
\overline f
\longrightarrow
\widetilde f_i
\longrightarrow
\ell_{ij}
\longrightarrow
A.
```

That is a meaningful representation change. It means the topology is not solely
a function of local pairwise features. The paper's displayed method should be
distinguished from such implementation-specific preprocessing unless the
version being documented explicitly includes it.

## 20. Permutation equivariance

Let P be a permutation of patch order. Transform feature rows:

```math
F'
=
PF.
```

Head and tail rows transform as:

```math
H'
=
PH,
\qquad
T'
=
PT.
```

The pairwise score matrix transforms by:

```math
L'
=
PLP^{\mathsf T}.
```

Top-k support is reindexed:

```math
\mathcal N_K'(\pi(i))
=
\{\pi(j):j\in\mathcal N_K(i)\}.
```

Edge embeddings transform with their endpoint pairs:

```math
r_{\pi(i)\pi(j)}'
=
r_{ij}.
```

The knowledge-aware weights and updated heads reindex:

```math
\pi_{\pi(i)\pi(j)}'
=
\pi_{ij},
\qquad
h_{\pi(i)}^{+\prime}
=
h_i^{+}.
```

Mean and max readouts are invariant:

```math
\mathcal R(PH^{+})
=
\mathcal R(H^{+}).
```

Thus WiKG is invariant to storage order while remaining sensitive to the
learned directed relation object.

## 21. Directional support versus undirected graph

An undirected graph would impose:

```math
A_{ij}
=
A_{ji}.
```

WiKG instead permits:

```math
A_{ij}
\ne
A_{ji}.
```

This matters because head i and tail j encode different roles. The message
from j into i is:

```math
\pi_{ij}t_j,
```

whereas the reverse message is:

```math
\pi_{ji}t_i.
```

Even if both edges exist, they need not carry equal information:

```math
\pi_{ij}t_j
\ne
\pi_{ji}t_i.
```

The directed graph can express asymmetric influence patterns, but the
asymmetry is learned from feature projections rather than supplied as a
biological causal direction.

## 22. Relation to coordinate graphs

Patch-GCN uses:

```math
A_{\mathrm{coord}}
=
\Gamma(C).
```

WiKG uses:

```math
A_{\mathrm{WiKG}}
=
\Gamma(F;W_h,W_t).
```

A distant pair can be connected in WiKG:

```math
\|c_i-c_j\|_2
\text{ large},
\qquad
j\in\mathcal N_K(i),
```

if its head-tail compatibility is high. A nearby pair can be excluded if its
learned relation score is low.

This increases interaction flexibility but removes the explicit spatial prior.
The learned topology can encode phenotype similarity, scanner artifacts,
sampling density, or label shortcuts.

## 23. Knowledge-aware attention versus ordinary GAT

Ordinary graph attention might use:

```math
u_{ij}^{\mathrm{GAT}}
=
a
\left(
Wh_i,
Wh_j
\right).
```

WiKG uses a triplet-aware score:

```math
u_{ij}^{\mathrm{WiKG}}
=
t_j^{\mathsf T}
\tanh
\left(
h_i+r_{ij}
\right).
```

The edge relation is a learned high-dimensional state:

```math
r_{ij}
=
\omega_{ij}t_j
+
(1-\omega_{ij})h_i.
```

If r-ij is removed, the model loses the explicit head-tail-edge triplet
interaction. It then becomes a different neighbor attention architecture.

## 24. What survives the local context operator

The updated head is:

```math
h_i^{+}
=
\Psi
\left(
h_i,
\sum_{j\in\mathcal N_K(i)}
\pi_{ij}t_j
\right).
```

The original candidate set and discarded tails are not retained individually
unless their information is encoded in the weighted sum.

The local bottleneck is:

```math
\{(h_i,r_{ij},t_j):j\in\mathcal N_K(i)\}
\longrightarrow
h_{\mathcal N(i)}.
```

The global bottleneck is:

```math
\{h_i^{+}\}_{i=1}^{N}
\longrightarrow
z.
```

With mean readout, the final surviving statistic is a first moment of
relation-contextualized heads. With max readout, it is a coordinatewise extreme.

## 25. Dynamic-topology collision

Two slides can have identical patch features but different projection
parameters during training, producing different supports:

```math
F_A=F_B,
\qquad
(W_h,W_t)_A
\ne
(W_h,W_t)_B
\Longrightarrow
A_A\ne A_B.
```

At inference with shared parameters, two slides can have different topologies
because their features differ:

```math
F_A\ne F_B
\Longrightarrow
A_A\ne A_B.
```

The support itself is part of the learned representation. A graph-level
explanation that shows only updated node features hides this discrete topology
choice.

## 26. Top-k instability counterexample

Consider one head with three candidate tails and k equal to one:

```math
\ell_{i1}=1.00,
\qquad
\ell_{i2}=0.99,
\qquad
\ell_{i3}=-1.00.
```

The selected support is:

```math
\mathcal N_1(i)=\{1\}.
```

A perturbation changes the scores to:

```math
\ell_{i1}'=0.98,
\qquad
\ell_{i2}'=1.01,
\qquad
\ell_{i3}'=-1.00.
```

Now:

```math
\mathcal N_1'(i)=\{2\}.
```

The support changed despite a small score perturbation. The relation embedding
and knowledge-aware message can change by an amount controlled by t-one versus
t-two, not by the size of the score perturbation alone.

## 27. Rare witness versus distributed signal

With mean graph readout:

```math
z^{\mathrm{mean}}
=
\frac{1}{N}
\sum_i h_i^{+},
```

a rare updated head can be diluted. Max readout retains a coordinatewise
extreme but can select an artifact.

WiKG's local top-k and knowledge attention can preserve a rare tail if it is
selected and receives high pi-ij. It can still fail if the patch is excluded at
the hard topology step:

```math
j\notin\mathcal N_K(i)
\Longrightarrow
t_j
\text{ cannot directly contribute to }h_{\mathcal N(i)}.
```

No later attention weight can recover a tail that was not selected.

## 28. Interpretability boundary

A high relation score ell-ij means the head-tail projections are compatible
enough to enter the candidate support. A high knowledge score u-ij means the
tail is favored after relation conditioning. A high global readout contribution
means the updated head affects the graph embedding.

These are three different quantities:

```math
\ell_{ij}
\quad
\text{topology score},
```

```math
\pi_{ij}
\quad
\text{local knowledge-aware routing weight},
```

```math
\Delta_i^{(q)}
=
q(G)-q(G_{-i})
\quad
\text{graph-aware output deletion}.
```

They should not be rendered as one undifferentiated heatmap.

For a full support-aware deletion, recompute:

```math
G_{-i}
=
\Gamma_{\mathrm{WiKG}}
\left(
F_{-i};W_h,W_t
\right).
```

Deleting a patch changes candidate scores and potentially the support of many
other heads. Holding A fixed tests a different intervention.

## 29. Complexity

The dense head-tail score matrix has cost:

```math
\mathcal O(N^2d).
```

Top-k extraction produces:

```math
|E|
=
Nk.
```

The local knowledge-aware attention then costs approximately:

```math
\mathcal O(Nkd).
```

The graph update and mean readout are linear in N for fixed k and d:

```math
\mathcal O(Nkd+Nd).
```

The dense pairwise construction is the important scaling bottleneck. Sparse
message aggregation after top-k does not eliminate the cost of computing all
head-tail candidate scores unless approximate retrieval or blockwise search is
used.

## 30. C/R/G/S placement

| Component | WiKG instantiation |
| --- | --- |
| C | head-tail top-k support, directed edge interpolation, triplet knowledge-aware softmax, dual additive/multiplicative interaction |
| R | global mean or coordinatewise max over updated head features |
| G | learned directed feature geometry; optional implementation-level slide-mean injection |
| S | slide classification with softmax cross-entropy |

The full composition is:

```math
F
\xrightarrow{
W_h,W_t
}
(H,T)
\xrightarrow{
\mathrm{TopK}(H T^{\mathsf T})
}
A
\xrightarrow{
r_{ij}=\omega_{ij}t_j+(1-\omega_{ij})h_i
}
R
\xrightarrow{
\mathrm{KAA}
}
H^{+}
\xrightarrow{
\mathrm{mean/max}
}
z
\xrightarrow{
\mathrm{softmax\ head}
}
\widehat Y.
```

The relation object is learned before the global readout. The graph is not a
fixed coordinate scaffold.

## 31. Sanity checks

### 31.1 Directedness check

Compare supports in both directions:

```math
j\in\mathcal N_K(i)
\quad
\text{and}
\quad
i\in\mathcal N_K(j).
```

The implementation should not silently symmetrize them.

### 31.2 Top-k recomputation

Perturb one patch feature and recompute all supports:

```math
A(F)
\quad
\text{versus}
\quad
A(F+\delta F).
```

Measure edge-set stability:

```math
\mathrm{Jaccard}
\left(
E(F),E(F+\delta F)
\right).
```

### 31.3 Edge ablation

Set relation states to a common baseline:

```math
r_{ij}
=
r_0.
```

Compare the output with the original relation-aware model. This tests whether
edge embeddings actually affect KAA.

### 31.4 KAA ablation

Replace u-ij with a head-tail-only score:

```math
u_{ij}^{\mathrm{ablate}}
=
t_j^{\mathsf T}\tanh(h_i).
```

The difference isolates the edge-conditioned path.

### 31.5 Mean versus max readout

Compare:

```math
z^{\mathrm{mean}}
\quad
\text{and}
\quad
z^{\mathrm{max}}.
```

A performance or explanation change is a readout effect, not necessarily a
topology effect.

### 31.6 Node-order equivariance

Permute features and verify:

```math
F'
=
PF,
\qquad
z(F')
=
z(F).
```

The support and edge embeddings should reindex consistently.

### 31.7 Support-aware deletion

Delete a node and rebuild the graph. Compare with a fixed-support deletion:

```math
q(G)-q(G_{-i}^{\mathrm{rebuild}})
\quad
\text{versus}
\quad
q(G)-q(G_{-i}^{\mathrm{fixed\ support}}).
```

The gap measures topology-mediated influence.

## 32. Bottom line

WiKG is a dynamic directed graph-context operator followed by a simple graph
readout:

```math
\boxed{
\mathrm{WiKG}
=
\mathrm{Readout}
\circ
\mathrm{DualInteraction}
\circ
\mathrm{KnowledgeAttention}
\circ
\mathrm{DirectedTopK}
\circ
\mathrm{HeadTailProjection}
}
```

Its paper-specific equations are:

```math
h_i=W_hf_i,
\qquad
t_i=W_tf_i,
```

```math
\mathcal N_K(i)
=
\mathrm{TopK}_j
\left(
h_i^{\mathsf T}t_j
\right),
```

```math
r_{ij}
=
\omega_{ij}t_j
+
(1-\omega_{ij})h_i,
```

```math
u_{ij}
=
t_j^{\mathsf T}
\tanh(h_i+r_{ij}),
```

```math
\pi_{ij}
=
\frac{\exp(u_{ij})}
{\sum_{r\in\mathcal N_K(i)}\exp(u_{ir})},
```

```math
h_{\mathcal N(i)}
=
\sum_{j\in\mathcal N_K(i)}
\pi_{ij}t_j,
```

```math
h_i^{+}
=
\sigma_1(W_1(h_i+h_{\mathcal N(i)}))
+
\sigma_2(W_2(h_i\odot h_{\mathcal N(i)})).
```

The surviving statistic is a mean or max of updated head features. The main
failure modes are top-k discontinuity, dense pairwise cost, learned topology
shortcuts, directional support instability, and global readout loss.

The C/R/G/S summary is:

```math
\boxed{
\text{learned directed support}
+
\text{edge-conditioned neighbor first moment}
+
\text{dual head-neighbor interaction}
+
\text{mean/max slide readout}
}
```

The correct question is not whether WiKG “finds the important patches.” It is:

```math
\text{which directed head-tail relations survive top-k selection, KAA, and the
final graph readout?}
```
