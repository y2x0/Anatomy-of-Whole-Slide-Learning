# Hybrid MIL Failure Modes and Counterexamples

A hybrid MIL model composes multiple context and readout operators:

```math
F_\theta
=
\mathcal H
\circ
\mathcal R
\circ
\mathcal C_K
\circ
\mathcal C_{K-1}
\circ
\cdots
\circ
\mathcal C_1
\circ
\Gamma_{\mathrm G}.
```

Examples include graph context followed by attention, transformer context
followed by hierarchical [CLS] readout, state-space context followed by MIL
attention, and cell graph context followed by tissue hierarchy.

The central failure is compositional:

```math
\boxed{
\text{a valid local operator can become invalid when its output is treated as
the input geometry or statistic of another operator}
}
```

The analysis must therefore identify each intermediate object and each
intermediate bottleneck.

## 1. Source anchors

The mapped hybrid examples are:

| Method or family | Source note | Hybrid composition |
| --- | --- | --- |
| Patch-GCN | [graph coordinate context](../graph_mil/02_patch_gcn_coordinate_context_and_survival.md) | coordinate graph context then global attention |
| WiKG | [dynamic graph readout](../graph_mil/04_wikg_dynamic_support_and_knowledge_readout.md) | learned directed context then mean or max |
| HACT | [hierarchical graph boundary](../hierarchical_mil/05_hact_and_hierarchical_graph_mil_boundary.md) | cell graph, assignment, tissue graph |
| HIPT | [nested transformer tokens](../hierarchical_mil/04_hipt_nested_tokens_and_multiscale_readout.md) | transformer context and multiscale [CLS] |
| DTFD-MIL | [two-tier distillation](../hierarchical_mil/03_dtfd_pseudo_bags_and_two_tier_distillation.md) | pseudo-bag selection and parent-slide attention |
| MambaMIL | [MambaMIL head](../state_space_mil/05_mambamil_vanilla_scan_and_mil_head.md) | state-space context then attention weighted mean |
| TransMIL | [transformer MIL context](../transformer_mil/02_transmil_self_attention_and_nystrom_approximation.md) | self-attention context and transformer bag readout |

The failure arguments generalize across these compositions; they do not claim
that every mapped paper implements every possible hybrid.

## 2. Intermediate-object notation

Let the initial patch field be H-0. A hybrid stack creates:

```math
H^{(1)}
=
\mathcal C_1(H^{(0)},G_1),
```

```math
H^{(2)}
=
\mathcal C_2(H^{(1)},G_2(H^{(1)})),
```

```math
z
=
\mathcal R(H^{(2)},G_3).
```

The second geometry can depend on the first contextual field. For example, a
feature graph built after an encoder or a dynamic top-k graph built from
head-tail projections is not independent of C-1.

The complete graph/geometry path is:

```math
G_1
=
\Gamma_1(H^{(0)},P),
\qquad
G_2
=
\Gamma_2(H^{(1)},P,\tau),
\qquad
z
=
\mathcal R(H^{(2)},G_2).
```

A hybrid failure can enter through:

```text
the initial object
the first geometry
the first context
the second geometry induced by context
the intermediate readout
the final readout
the task supervision
```

## 3. Readout invariance does not repair context dependence

Let R be a permutation-invariant mean and C an order-sensitive recurrence:

```math
\mathcal R(PY)
=
\mathcal R(Y),
```

```math
\mathcal C(PH)
\ne
P\mathcal C(H).
```

Then:

```math
\mathcal R
\left(
\mathcal C(PH)
\right)
\ne
\mathcal R
\left(
\mathcal C(H)
\right)
```

in general.

The same applies to graph context. A permutation-invariant graph readout
removes node-index dependence but does not make two adjacency matrices
equivalent:

```math
\mathcal R
\left(
\mathcal C(H,A)
\right)
\ne
\mathcal R
\left(
\mathcal C(H,A')
\right)
```

when A and A-prime produce different contextual states.

This is the first hybrid counterexample: a symmetric final operator cannot undo
an asymmetric operator that acted before it.

## 4. Context operators do not generally commute

Let C-G be graph context and C-S be state-space context. In general:

```math
\mathcal C_{\mathrm S}
\left(
\mathcal C_{\mathrm G}(H,A),\sigma
\right)
\ne
\mathcal C_{\mathrm G}
\left(
\mathcal C_{\mathrm S}(H,\sigma),A
\right).
```

Graph context changes node features before the state-space order or scan
geometry is interpreted. State-space context changes features before feature
graph construction or attention scores are computed.

A special equality requires a commutation relation:

```math
\mathcal C_{\mathrm S}
\circ
\mathcal C_{\mathrm G}
=
\mathcal C_{\mathrm G}
\circ
\mathcal C_{\mathrm S}.
```

Nonlinear normalization, gating, support selection, and top-k operations make
this equality exceptional.

## 5. Coarsening and context do not commute

Let B be a child-parent assignment and A a fine graph. Graph-first followed by
sum transfer gives:

```math
H_{\mathrm{parent}}^{\mathrm{graph-first}}
=
B^{\mathsf T}
\widetilde A H.
```

Coarsen-first followed by a parent operator gives:

```math
H_{\mathrm{parent}}^{\mathrm{coarse-first}}
=
A_{\mathrm P}
B^{\mathsf T}H.
```

They are equal only if:

```math
B^{\mathsf T}\widetilde A
=
A_{\mathrm P}B^{\mathsf T}.
```

Edges crossing parent boundaries violate this identity in general. A graph-first
model can transfer a cross-boundary signal before coarsening; a coarse-first
model may remove the child identity first.

The distinction applies to HACT, graph-hierarchy hybrids, and any region
pipeline that performs graph context before or after pooling.

## 6. Exact two-node non-commutativity example

Let:

```math
H
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
A
=
\frac{1}{2}
\begin{bmatrix}
1&1\\
1&1
\end{bmatrix},
\qquad
B
=
\begin{bmatrix}
1\\
1
\end{bmatrix}.
```

Graph-first and sum transfer give:

```math
B^{\mathsf T}AH
=
1.
```

If sum transfer occurs first, the parent receives one scalar. A parent context
cannot reconstruct whether the mass came from the first node, the second node,
or both:

```math
B^{\mathsf T}
\begin{bmatrix}
1\\
0
\end{bmatrix}
=
B^{\mathsf T}
\begin{bmatrix}
0\\
1
\end{bmatrix}
=
1.
```

The graph-first path can use A before this collision; the coarse-first path
cannot.

## 7. Dynamic geometry after context

Suppose a feature graph is constructed from contextual states:

```math
A^{(2)}
=
\Gamma_{\mathrm{kNN}}
\left(
H^{(1)}
\right),
\qquad
H^{(1)}
=
\mathcal C_1(H^{(0)},A^{(1)}).
```

A perturbation of one initial patch affects both the first contextual field and
the second support:

```math
\frac{\partial z}{\partial h_k^{(0)}}
=
\frac{\partial z}{\partial H^{(2)}}
\frac{\partial H^{(2)}}{\partial A^{(2)}}
\frac{\partial A^{(2)}}{\partial H^{(1)}}
\frac{\partial H^{(1)}}{\partial h_k^{(0)}}
+
\frac{\partial z}{\partial H^{(2)}}
\frac{\partial H^{(2)}}{\partial h_k^{(0)}}.
```

For hard kNN or top-k support, the derivative does not capture discrete
neighbor replacement. A small feature perturbation can alter the next graph
before the final readout is applied.

## 8. Double counting through multiple paths

Let a residual graph layer be:

```math
\widetilde H
=
(I+\widetilde A)HW.
```

A patch contributes to its own state and to neighboring states. A final sum
then gives:

```math
z
=
\mathbf 1^{\mathsf T}
(I+\widetilde A)HW.
```

The coefficient of input node v is proportional to the column sum:

```math
c_v
=
\sum_u
\left[
I+\widetilde A
\right]_{uv}.
```

If column sums vary with degree, identical patch features can receive different
global weight because of graph position. With multiple graph layers:

```math
z
=
\mathbf 1^{\mathsf T}
\left(
\prod_{\ell=1}^{L}
(I+\widetilde A_\ell)
\right)
HW.
```

A high-degree node can be counted through many paths. Mean normalization changes
scale but does not necessarily remove path multiplicity when context states
depend on degree.

## 9. Attention collapse after context

Let attention readout score contextualized states:

```math
\alpha_j
=
\frac{\exp(g(\widetilde h_j))}
{\sum_k\exp(g(\widetilde h_k))}.
```

If contextualization collapses states:

```math
\widetilde h_j
\approx
\widetilde h_k
\qquad
\forall j,k,
```

then scores tend to be similar and:

```math
\alpha_j
\approx
\frac{1}{N}.
```

The opposite failure is a dominant score:

```math
\max_j\alpha_j
\approx
1.
```

Uniform attention hides localization; dominant attention creates brittle
selection. The score entropy is:

```math
H(\alpha)
=
-\sum_j\alpha_j\log\alpha_j.
```

The effective support is:

```math
N_{\mathrm{eff}}
=
\exp(H(\alpha)).
```

These values should be compared with deletion effects, not treated as
explanations by themselves.

## 10. Hybrid explanation chain rule

Let:

```math
z
=
\mathcal R
\left(
\widetilde H
\right),
\qquad
\widetilde H
=
\mathcal C
\left(
H,A,\sigma,B
\right).
```

The total derivative is:

```math
\frac{\partial z}{\partial h_k}
=
\sum_j
\frac{\partial z}{\partial\widetilde h_j}
\frac{\partial\widetilde h_j}{\partial h_k}
+
\frac{\partial z}{\partial A}
\frac{\partial A}{\partial h_k}
+
\frac{\partial z}{\partial\sigma}
\frac{\partial\sigma}{\partial h_k}
+
\frac{\partial z}{\partial B}
\frac{\partial B}{\partial h_k}.
```

Hard graph support, discrete sequence order, and hard assignment maps make
some terms set-valued or undefined at boundaries. An attention map at R shows
only the first factor.

The full deletion intervention is:

```math
\Delta_k^{\mathrm{full}}
=
q(H,A,\sigma,B)
-
q(H\setminus k,A',\sigma',B').
```

The primed geometry must be specified. Freezing A, sigma, or B produces a
conditional intervention, not the full model effect.

## 11. Geometry mismatch in hybrid stacks

Different geometry operators impose different equivalence relations:

```text
coordinate graph:
    near in physical space

feature graph:
    similar under encoder geometry

sequence scan:
    near in selected order

hierarchy:
    shares a parent

transformer:
    globally available within its token support
```

A hybrid can therefore use incompatible neighborhoods. A coordinate graph may
send a message across a tissue boundary while a feature graph later connects
morphologically similar distant patches. A sequence branch may place those
patches far apart or adjacent for unrelated reasons.

A controlled geometry test holds H fixed and changes one relation at a time:

```math
\Delta_{\mathrm G_1}
=
d(F(H,G_1,G_2),F(H,G_1',G_2)),
```

```math
\Delta_{\mathrm G_2}
=
d(F(H,G_1,G_2),F(H,G_1,G_2')).
```

Without this separation, a gain cannot be assigned to a particular context
operator.

## 12. Bottleneck migration

A hybrid can move the computational bottleneck:

```math
O(N^2d)
\longrightarrow
O(Ed^2)
+
O(NKd^2)
+
O(R^2d).
```

A sparse graph may reduce pairwise attention but incur graph construction cost.
A hierarchy may reduce local attention but create a large parent count. A
state-space branch may be linear in N while its feature projection, padding, or
reordering dominates memory.

The total cost is:

```math
\mathrm{Cost}_{\mathrm{total}}
=
\mathrm{Cost}_{\mathrm G}
+
\sum_\ell
\mathrm{Cost}_{\mathrm C_\ell}
+
\sum_\ell
\mathrm{Cost}_{\mathrm R_\ell}.
```

Reporting only the asymptotic cost of one module is insufficient.

## 13. Order and graph mismatch

Suppose a sequence branch uses order sigma and a graph branch uses adjacency A.
A patch pair can be:

```math
|\,\sigma(u)-\sigma(v)\,|
\text{ small}
\qquad
\text{but}
\qquad
A_{uv}=0,
```

or:

```math
A_{uv}=1
\qquad
\text{but}
\qquad
|\,\sigma(u)-\sigma(v)\,|
\text{ large}.
```

The two branches then assign different context strength to the same pair. A
fusion operator can average, concatenate, or gate these incompatible signals,
but it does not resolve which geometry is valid.

A disagreement map is:

```math
D_{uv}
=
\mathbf 1\{A_{uv}=1\}
-
\mathbf 1
\left\{
|\sigma(u)-\sigma(v)|\le w
\right\}.
```

Summarize D by tissue type, physical distance, and task contribution.

## 14. Hierarchical selection amplifies graph errors

Let graph context produce child states:

```math
\widetilde H
=
\widetilde A H.
```

Let a parent selector choose one child per parent:

```math
h_b^{\mathrm{parent}}
=
\widetilde h_{j_b},
\qquad
j_b
=
\underset{j\in V_b}{\arg\max}
s_j.
```

A small graph perturbation can change s-j and therefore the selected child.
The final parent field changes discontinuously:

```math
j_b\ne j_b'
\Longrightarrow
h_b^{\mathrm{parent}}
\ne
h_b^{\mathrm{parent}\prime}
```

even when all contextual states changed continuously.

DTFD MaxS and MAS are examples of sparse boundary selection. A graph or
transformer before selection changes which child appears to be maximal.

## 15. Transformer context and [CLS] collisions

For a token matrix X:

```math
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
```

Self-attention is:

```math
\mathrm{Attn}(X)
=
\mathrm{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt d}
\right)V.
```

A [CLS] readout is a learned projection of the entire contextualized token
field. Two different fields can collide:

```math
X\ne X',
\qquad
\mathrm{CLS}
\left(
\mathrm{Transformer}(X)
\right)
=
\mathrm{CLS}
\left(
\mathrm{Transformer}(X')
\right).
```

If the [CLS] vector feeds a graph or state-space branch, the downstream branch
cannot recover the lost token distinction. Hybrid depth does not reverse a
many-to-one boundary.

## 16. State-space context and attention readout

For a causal recurrence:

```math
s_t
=
\overline A_t s_{t-1}
+
\overline B_tu_t,
\qquad
y_t=C_ts_t+D_tu_t.
```

Attention readout is:

```math
z
=
\sum_t\alpha_ty_t.
```

The derivative with respect to input k is:

```math
\frac{\partial z}{\partial u_k}
=
\sum_{t\ge k}
\left(
\frac{\partial\alpha_t}{\partial u_k}y_t
+
\alpha_t\frac{\partial y_t}{\partial u_k}
\right).
```

The weight alpha-t cannot be interpreted as input-k importance because y-t
contains a prefix state and alpha-t itself depends on contextualized output.

This is the MambaMIL-specific distinction between released attention readout and
state-space memory.

## 17. Counterexample: same attention map, different graph

Let two graphs have the same node states and the same final attention scores:

```math
\widetilde H=\widetilde H',
\qquad
\alpha=\alpha'.
```

If the contextualization graphs differ but happen to produce the same current
states, a future deletion or perturbation can differ:

```math
q(G,\widetilde H)
=
q(G',\widetilde H'),
\qquad
q(G\setminus v)
\ne
q(G'\setminus v).
```

The final attention map is identical, but the dependency structure is not.
This is why attention maps cannot identify graph message paths.

## 18. Counterexample: same coarse statistic, different fine arrangement

Let two parent groups have identical means:

```math
\bar h_b
=
\bar h_b'
\qquad
\forall b.
```

A parent-level attention head receives identical parent states and must collide,
even if child arrangements differ:

```math
\{h_a:a\in V_b\}
\ne
\{h_a':a\in V_b'\}.
```

A child-level graph or second-moment skip would be required to distinguish them.

## 19. Counterexample: graph-first versus hierarchy-first

Let a fine graph contain one cross-boundary edge connecting child one in parent
one to child two in parent two. Let B assign the children to their parents.

Graph-first transmits the edge before summation:

```math
B^{\mathsf T}AH
\ne
B^{\mathsf T}H.
```

Hierarchy-first collapses each parent before a parent graph is built. If the
parent graph does not include the cross-boundary edge or its sufficient statistic,
the interaction is lost.

The two models can have identical parameter counts and different predictions
because they implement different operator order.

## 20. Counterexample: sum double count

Let all input node states equal u and use one residual graph layer:

```math
\widetilde H
=
(I+A)
\mathbf 1u^{\mathsf T}.
```

A sum readout gives:

```math
z
=
\mathbf 1^{\mathsf T}
(I+A)
\mathbf 1u^{\mathsf T}.
```

The coefficient is:

```math
\mathbf 1^{\mathsf T}
(I+A)
\mathbf 1
=
N+\sum_{u,v}A_{uv}.
```

Two slides with equal morphology but different edge counts produce different
representations. Mean can reduce scale but not eliminate topology-dependent
state changes.

## 21. Counterexample: two orders, one fused vector

Let a two-token scalar sequence have order-sensitive outputs:

```math
Y^{(1)}
=
\begin{bmatrix}
u_1\\
au_1+u_2
\end{bmatrix},
\qquad
Y^{(2)}
=
\begin{bmatrix}
u_2\\
au_2+u_1
\end{bmatrix}.
```

A fused mean can collide for particular a and inputs even though each branch
differs. A fused readout therefore does not prove order invariance; it may
simply hide branch disagreement.

## 22. Supervision mismatch

Let a hybrid have intermediate losses:

```math
\mathcal L
=
\mathcal L_{\mathrm{slide}}
+
\lambda_1\mathcal L_{\mathrm{child}}
+
\lambda_2\mathcal L_{\mathrm{parent}}.
```

Changing lambda values changes which information survives boundaries. If the
child target rewards local morphology but the slide target rewards a global
shortcut, the gradients can oppose:

```math
\left\langle
\nabla_\theta\mathcal L_{\mathrm{slide}},
\nabla_\theta\mathcal L_{\mathrm{child}}
\right\rangle
<0.
```

A hybrid architecture does not have one fixed surviving statistic across
training objectives.

## 23. Deployment shift compounds across operators

Let training and test distributions differ in any geometry or intermediate
object:

```math
P_{\mathrm{train}}(H,G_1,G_2)
\ne
P_{\mathrm{test}}(H,G_1,G_2).
```

Even if patch features are stable, a shift in coordinate graph, sequence order,
parent occupancy, or pseudo-type distribution can change downstream states.

A factorized shift audit measures:

```math
\Delta_{G_\ell}
=
d
\left(
P_{\mathrm{train}}(G_\ell),
P_{\mathrm{test}}(G_\ell)
\right),
```

```math
\Delta_{H_\ell}
=
d
\left(
P_{\mathrm{train}}(H^{(\ell)}),
P_{\mathrm{test}}(H^{(\ell)})
\right).
```

A later operator can amplify a small early shift through support selection or
normalization.

## 24. Hybrid intervention suite

### 24.1 Operator permutation

Swap two context operators while preserving their parameters:

```math
F_{12}(H)
=
\mathcal C_2(\mathcal C_1(H)),
\qquad
F_{21}(H)
=
\mathcal C_1(\mathcal C_2(H)).
```

Measure their difference and record whether geometry inputs must also change.

### 24.2 Geometry replacement

Hold features fixed and replace one geometry:

```math
F(H,G_1,G_2)
\quad\longrightarrow\quad
F(H,G_1',G_2).
```

Do not change all graphs simultaneously if the goal is attribution.

### 24.3 Boundary readout swap

Hold each contextual field fixed and replace R with sum, mean, max, attention,
or [CLS]. This isolates surviving statistic from context.

### 24.4 Branch ablation

For multi-branch state-space or graph/transformer hybrids, evaluate each branch
and every fusion combination.

### 24.5 Full deletion

Remove a patch and rebuild every affected graph, order, parent map, context
state, and readout score. Compare with fixed-geometry deletion.

### 24.6 Count and density matching

Match node count, edge count, parent count, and valid-token count across
groups before interpreting task performance.

## 25. C/R/G/S failure matrix

| Hybrid | C failure | R failure | G failure | S/deployment failure |
| --- | --- | --- | --- | --- |
| graph then attention | smoothing or wrong message paths | weighted moment hides topology and multiplicity | coordinate or feature support mismatch | density shortcut and attention misinterpretation |
| transformer then hierarchy | [CLS] bottleneck before coarse context | parent tokens collide | window boundaries and positional scale | pretraining/downstream scale shift |
| state-space then attention | order-conditioned finite memory | score map is not patch effect | arbitrary or acquisition order | order shortcut and length shift |
| graph then hierarchy | cross-boundary message mismatch | assignment nullspace and double count | wrong parent map | segmentation and occupancy shift |
| pseudo-bag then attention | inherited local label noise | selection loses unchosen patches | random grouping is not tissue geometry | pseudo-bag sampling and class imbalance |
| multi-order fusion | branch cancellation or redundancy | fused vector hides disagreement | selected orders remain biased | branch-specific cohort shift |

## 26. C/R/G/S reporting rule

A hybrid MIL paper should specify:

```text
1. the initial slide object
2. every geometry construction and whether it depends on contextual states
3. every context operator and its input/output shape
4. every child-to-parent or token readout
5. branch fusion and residual paths
6. the surviving statistic after each boundary
7. intermediate and final supervision
8. operator ordering and non-commuting alternatives
9. graph, order, parent, count, and padding conventions
10. full deletion and branch ablation tests
```

The compact signature is:

```math
\boxed{
[
\text{object}
\mid
G_1,C_1,R_1
\mid
G_2,C_2,R_2
\mid
\cdots
\mid
S
]
}
```

## 27. Bottom line

Hybrid MIL is not “all modules at once.” It is an ordered composition:

```math
\boxed{
\text{object}
\longrightarrow
\text{geometry}
\longrightarrow
\text{context}
\longrightarrow
\text{boundary statistic}
\longrightarrow
\text{new geometry}
\longrightarrow
\text{task head}
}
```

The primary risk is that each operator changes the object expected by the next
one. A final attention map cannot explain a graph path, a graph readout cannot
restore a discarded [CLS] token, and a symmetric final pooling cannot remove
order dependence created by a scan.

The correct question is:

```math
\boxed{
\text{which distinctions survive every non-commuting operator in the hybrid
composition, and which supervision makes those distinctions useful?}
}
```
