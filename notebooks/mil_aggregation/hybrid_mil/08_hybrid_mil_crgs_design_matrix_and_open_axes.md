# Hybrid MIL C/R/G/S Design Matrix and Open Axes

Hybrid MIL is an ordered composition of slide representations. Let the initial
patch object be H-0 and let each stage expose its own geometry, context, and
readout:

```math
H^{(r+1)}
=
\mathcal R_r
\left(
\mathcal C_r
\left(
H^{(r)},G_r
\right),
B_r
\right),
\qquad
r=0,\ldots,R-1.
```

The task output is:

```math
z
=
\mathcal R_{\mathrm{slide}}
\left(
H^{(R)},G_R
\right),
\qquad
\widehat y
=
\mathcal H(z).
```

The hybrid signature is:

```math
\boxed{
[
\text{object}
\mid
G_0,C_0,R_0
\mid
G_1,C_1,R_1
\mid
\cdots
\mid
G_R,C_R,R_{\mathrm{slide}}
\mid
S
]
}
```

The operators need not commute. Their order is part of the inductive bias.

## 1. What the four axes mean in a hybrid

### Geometry G

G specifies the relation or parent object at each stage:

```math
G_r
=
\Gamma_r
\left(
H^{(r)},P_r,\tau_r,\phi_r
\right).
```

Possible objects include coordinate graphs, feature graphs, sequence orders,
nested windows, parent maps, and learned top-k support.

### Context C

C changes the representation before the next boundary:

```math
\widetilde H^{(r)}
=
\mathcal C_r
\left(
H^{(r)},G_r
\right).
```

Possible contexts include graph message passing, transformer self-attention,
state-space scans, gated attention, and cross-scale graph context.

### Readout R

R compresses a field or creates the next-level object:

```math
H^{(r+1)}
=
\mathcal R_r
\left(
\widetilde H^{(r)},B_r
\right).
```

Possible readouts include sum, mean, max, attention, [CLS], top-k selection,
prototype occupancy, and assignment transfer.

### Supervision S

S contains the task labels and intermediate objectives:

```math
S
=
\left(
\mathcal Y_{\mathrm{slide}},
\mathcal Y_{\mathrm{intermediate}},
\mathcal L_{\mathrm{task}},
\mathcal L_{\mathrm{aux}},
\mathcal D_{\mathrm{split}}
\right).
```

S determines which intermediate collisions are acceptable.

## 2. Source anchors

The matrix below uses the paper-specific derivations already in the repo:

| Method | Source note | Hybrid signature |
| --- | --- | --- |
| Patch-GCN | [coordinate graph survival](../graph_mil/02_patch_gcn_coordinate_context_and_survival.md) | coordinate graph context then global attention |
| HEAT | [typed graph readout](../graph_mil/03_heat_typed_context_and_pseudo_label_readout.md) | edge-conditioned typed context then PL rows |
| WiKG | [dynamic graph readout](../graph_mil/04_wikg_dynamic_support_and_knowledge_readout.md) | directed top-k context then mean or max |
| HACT | [hierarchical graph boundary](../graph_mil/05_hact_and_hierarchical_graph_mil_boundary.md) | cell context, assignment transfer, tissue context |
| DTFD-MIL | [two-tier distillation](../hierarchical_mil/03_dtfd_pseudo_bags_and_two_tier_distillation.md) | pseudo-bag selection then parent attention |
| HIPT | [nested tokens](../hierarchical_mil/04_hipt_nested_tokens_and_multiscale_readout.md) | transformer context and multiscale [CLS] |
| S4MIL | [S4MIL readout](../state_space_mil/03_s4mil_structured_state_space_readout.md) | structured state-space context then max |
| MambaMIL | [MambaMIL head](../state_space_mil/05_mambamil_vanilla_scan_and_mil_head.md) | selective scan then attention weighted mean |
| TransMIL | [Transformer MIL](../transformer_mil/02_transmil_self_attention_and_nystrom_approximation.md) | self-attention context then transformer bag summary |

## 3. Paper placement matrix

| Method | G: object | C: context | R: readout | Surviving statistic | S: task channel |
| --- | --- | --- | --- | --- | --- |
| Patch-GCN | coordinate kNN patch graph | local DeepGCN-style context | global attention | weighted first moment of contextual nodes | patient-level survival or slide task |
| HEAT | feature kNN, node types, Pearson edge attributes | heterogeneous edge-conditioned attention | type rows then graph readout | typed summaries | slide-level task with teacher attributes |
| WiKG | directed learned top-k graph | knowledge-aware relation context | mean or coordinatewise max | first moment or extrema | slide classification |
| HACT | cell graph, tissue graph, assignment | fine graph context and coarse graph context | assignment transfer then tissue readout | hierarchical coarse statistic | source region-level classification |
| DTFD-MIL | random pseudo-bags | Tier-1 and Tier-2 attention | selection or AFS then attention | selected candidates or nested weighted moments | inherited slide labels and final classification |
| HIPT | nested spatial windows | transformer context at three scales | [CLS] at every boundary | learned nonlinear multiscale tokens | DINO plus downstream slide task |
| S4MIL | inherited sequence/grid order | structured S4D context | coordinatewise max | channelwise extreme | slide and optional patch supervision |
| MambaMIL | selected sequence order | selective causal or bidirectional scan | attention weighted mean | weighted first moment of order-conditioned states | classification or discrete survival head |
| TransMIL | ordered bag plus PPEG/grid structure | Nystrom self-attention | transformer bag summary | learned global token statistic | slide classification |

The matrix separates context from readout. “Attention” can mean local context,
global pooling, or both; those are not the same operator.

## 4. Operator signatures

A graph-then-attention model has:

```math
F_{\mathrm{GA}}
=
\mathcal H
\circ
\mathcal R_{\mathrm{attn}}
\circ
\mathcal C_{\mathrm{graph}}
\circ
\Gamma_{\mathrm{coord}}.
```

A state-space-then-attention model has:

```math
F_{\mathrm{SA}}
=
\mathcal H
\circ
\mathcal R_{\mathrm{attn}}
\circ
\mathcal C_{\mathrm{SSM}}
\circ
P_\sigma.
```

A hierarchy-then-graph model has:

```math
F_{\mathrm{HG}}
=
\mathcal H
\circ
\mathcal R_{\mathrm{graph}}
\circ
\mathcal C_{\mathrm{parent}}
\circ
\mathcal R_{\mathrm{child}}
\circ
\mathcal C_{\mathrm{child}}
\circ
\Gamma_{\mathrm{child}}.
```

A graph-then-hierarchy model has:

```math
F_{\mathrm{GH}}
=
\mathcal H
\circ
\mathcal R_{\mathrm{parent}}
\circ
\mathcal C_{\mathrm{parent}}
\circ
\mathcal R_{\mathrm{child}}
\circ
\mathcal C_{\mathrm{graph}}
\circ
\Gamma_{\mathrm{graph}}.
```

The expressions are equal only under restrictive commutation conditions.

## 5. Same context, different readout

Fix contextualized states:

```math
\widetilde H
=
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix}.
```

Different readouts give:

```math
z^{\mathrm{sum}}=1,
\qquad
z^{\mathrm{mean}}=\frac{1}{3},
\qquad
z^{\mathrm{max}}=1.
```

For attention weights alpha:

```math
z^{\mathrm{attn}}
=
\alpha_1.
```

The context is identical. R changes the representation and therefore changes
the task head's hypothesis class.

## 6. Same readout, different context

Fix mean readout and two context fields:

```math
Y^{(1)}
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
Y^{(2)}
=
\begin{bmatrix}
1/2\\
1/2
\end{bmatrix}.
```

Both means equal one-half:

```math
\frac{1}{2}\mathbf 1^{\mathsf T}Y^{(1)}
=
\frac{1}{2}\mathbf 1^{\mathsf T}Y^{(2)}
=
\frac{1}{2}.
```

The node-level contexts differ even though the chosen R collides. A small
slide-level ablation does not prove C is irrelevant.

## 7. Context and readout non-commutativity

Let A be a graph propagation matrix and B a parent assignment. Graph-first
coarsening is:

```math
z_{\mathrm{graph-first}}
=
B^{\mathsf T}AH.
```

Coarsen-first parent context is:

```math
z_{\mathrm{coarse-first}}
=
A_{\mathrm P}B^{\mathsf T}H.
```

Equality requires:

```math
B^{\mathsf T}A
=
A_{\mathrm P}B^{\mathsf T}.
```

For a state-space scan and mean:

```math
z_{\mathrm{scan-first}}
=
\frac{1}{L}\mathbf 1^{\mathsf T}
\mathcal C_{\mathrm{SSM}}(P_\sigma H).
```

A mean-first model receives only one vector and cannot reconstruct the ordered
prefixes. The scan and mean do not commute.

## 8. Surviving-statistic composition matrix

| Composition | First statistic introduced | Final statistic | Main loss |
| --- | --- | --- | --- |
| graph then attention | relation-conditioned node field | weighted first moment | edge paths and multiplicity |
| transformer then [CLS] | pairwise token interactions | learned nonlinear summary | child field outside [CLS] image |
| state-space then mean | order-conditioned trajectory | first moment | positions and finite-state nullspace |
| state-space then attention | order-conditioned trajectory and scores | weighted first moment | score mixtures and prefix decomposition |
| hierarchy then attention | parent summaries | weighted parent moment | child details below boundary |
| graph then hierarchy | spatial context before assignment | coarse graph statistic | cross-boundary and within-parent contrasts |
| pseudo-bag selection then attention | sparse representatives | parent weighted moment | all unselected patches |
| dynamic graph then max | learned relational field | coordinatewise extreme | support multiplicity and edge triples |

“Weighted first moment” is not one universal object. Its support and the states
being averaged depend on all earlier C and R operators.

## 9. Geometry composition matrix

| First geometry | Second geometry | Potential compatibility | Main conflict |
| --- | --- | --- | --- |
| coordinates | feature kNN | local plus semantic relations | physical and feature neighborhoods disagree |
| coordinates | sequence order | raster or Hilbert may preserve locality | file order can be unrelated to space |
| graph support | hierarchy | graph messages can inform parent states | cross-boundary messages may be lost |
| hierarchy | graph support | parent graph can model coarse layout | child geometry was already compressed |
| sequence order | dynamic top-k | state-conditioned compatibility | order artifacts enter learned support |
| pseudo-bag partition | parent attention | virtual sample aggregation | pseudo-bag is not anatomical region |
| nested windows | global transformer | multiscale context | [CLS] boundaries create collisions |

A hybrid should state whether the second geometry is constructed from raw or
contextualized features.

## 10. Supervision composition

A hybrid objective can be:

```math
\mathcal L
=
\mathcal L_{\mathrm{slide}}
+
\sum_{r=0}^{R-1}
\lambda_r\mathcal L_r^{\mathrm{aux}}
+
\mathcal L_{\mathrm{reg}}.
```

For survival:

```math
\mathcal L_{\mathrm{slide}}
=
\mathcal L_{\mathrm{Cox}}
\quad
\text{or}
\quad
\mathcal L_{\mathrm{hazard}}.
```

For classification:

```math
\mathcal L_{\mathrm{slide}}
=
\mathcal L_{\mathrm{CE}}.
```

For foundation or contrastive pretraining:

```math
\mathcal L_{\mathrm{slide}}
=
\mathcal L_{\mathrm{contrastive}}
\quad
\text{or}
\quad
\mathcal L_{\mathrm{DINO}}.
```

The intermediate losses select which boundaries retain local information. A
hybrid with the same forward map but different lambda values is a different
trained representation.

## 11. Complexity matrix

Let N be patch count, E graph edges, K dynamic degree, R parent count, d width,
and L sequence length.

| Stage | Cost | Main risk |
| --- | --- | --- |
| coordinate or feature graph construction | index-dependent, often superlinear | memory and distribution shift |
| sparse graph context | order E d-squared | high degree or dense support |
| dense transformer context | order N-squared-d | quadratic attention |
| Nystrom transformer | order N m d plus landmark terms | approximation error and landmark choice |
| state-space scan | order L d N-state | state and projection constants |
| local hierarchy attention | order sum m-r-squared-d | unbalanced parents |
| parent attention | order R-squared-d | excessive parent count |
| top-k support | order N-squared-d before selection in dense candidate form | support construction |
| global attention readout | order N d plus score network | normalization and support |
| HACT assignment transfer | order nnz(B) d | fine-cell storage |

Total cost must include construction, context, readout, and fusion:

```math
\mathrm{Cost}_{\mathrm{total}}
=
\mathrm{Cost}_{\mathrm G}
+
\mathrm{Cost}_{\mathrm C}
+
\mathrm{Cost}_{\mathrm R}
+
\mathrm{Cost}_{\mathrm{fusion}}.
```

## 12. Information budget across hybrid stages

Let the dimension of each object be:

```math
H^{(r)}
\in
\mathbb R^{N_r\times d_r}.
```

A boundary changes the number of degrees of freedom. For a linear assignment
boundary, the Jacobian rank is:

```math
\mathrm{rank}
\left(
B_r^{\mathsf T}\otimes I_{d_r}
\right)
=
\mathrm{rank}(B_r)d_r.
```

For a final readout z in R-q, the overall Jacobian rank is at most q:

```math
\mathrm{rank}
\left(
\frac{\partial z}{\partial\mathrm{vec}(H^{(r)})}
\right)
\le q.
```

A hybrid can have a high-dimensional intermediate field and a small final
statistic. Depth does not imply that all intermediate information reaches the
task head.

## 13. Equivalence tests

### 13.1 Operator-order test

Compute:

```math
\Delta_{\mathrm{order}}
=
d
\left(
F_{\mathcal C_2\circ\mathcal C_1}(H),
F_{\mathcal C_1\circ\mathcal C_2}(H)
\right).
```

Record which geometries need to be rebuilt after the swap.

### 13.2 Geometry replacement

Hold H fixed:

```math
\Delta_{\mathrm G}
=
d
\left(
F(H,G),
F(H,G')
\right).
```

Replace one geometry at a time.

### 13.3 Readout replacement

Hold the contextual field fixed:

```math
\Delta_{\mathrm R}
=
d
\left(
\mathcal R(H),
\mathcal R'(H)
\right).
```

This isolates surviving statistic.

### 13.4 Branch ablation

For a two-branch composition:

```math
F_{\mathrm{fused}},
\qquad
F_{\mathrm{branch1}},
\qquad
F_{\mathrm{branch2}}.
```

The fused score can conceal cancellation or redundancy.

### 13.5 Full deletion

Remove a patch and rebuild all dependent objects:

```math
F(H,G(H))
\longrightarrow
F(H\setminus h_k,G(H\setminus h_k)).
```

Compare against fixed-geometry deletion to separate feature, context, and
geometry effects.

## 14. Design axes

An open design axis is meaningful only if it answers what information the new
composition should preserve.

### 14.1 Coordinate graph plus state-space context

Use a coordinate graph to define local neighborhoods and an SSM to process a
local or region order:

```math
H^{(1)}
=
\mathcal C_{\mathrm{graph}}(H,G_{\mathrm{coord}}),
```

```math
Y
=
\mathcal C_{\mathrm{SSM}}(P_\sigma H^{(1)}).
```

Question:

```math
\text{can the order be calibrated against physical graph distance?}
```

Measure both sequence-distance and physical-distance influence.

### 14.2 Graph context plus distribution readout

After graph context, retain per-region moments:

```math
z_b
=
\left[
\bar h_b
\middle\Vert
\mathrm{vec}(\widehat\Sigma_b)
\middle\Vert
\log(1+m_b)
\right].
```

Question:

```math
\text{does the distributional statistic preserve burden lost by mean attention?}
```

### 14.3 Hierarchy plus horizon-specific survival readout

Use:

```math
z_i^{(k)}
=
\mathcal R_k(H_i^{(L)}),
\qquad
\widehat h_i^{(k)}
=
\sigma(w_k^{\mathsf T}z_i^{(k)}+b_k).
```

Question:

```math
\text{which scale carries risk at horizon k, and does that scale remain stable?}
```

### 14.4 Transformer context plus top-k selection

Use self-attention to contextualize, then select candidates:

```math
j_r
=
\underset{j}{\arg\max}
s_j
\left(
\mathcal C_{\mathrm{Transformer}}(H)
\right).
```

Question:

```math
\text{does contextual selection preserve rare morphology or amplify attention
shortcuts?}
```

### 14.5 Learned routing plus prototype readout

Use routing probabilities and prototype assignments:

```math
\pi_{ab}
=
\mathrm{softmax}_b(r(h_a)),
```

```math
c_{bk}
=
\sum_a\pi_{ab}\rho_k(h_a).
```

Question:

```math
\text{can routing geometry and prototype occupancy be identified separately?}
```

## 15. Hybrid reporting matrix

A complete report should fill:

| Field | Required statement |
| --- | --- |
| initial object | set, sequence, graph, hierarchy, or distribution |
| G-0 | relation construction and invariance |
| C-0 | context equation and receptive field |
| R-0 | boundary statistic and shape |
| intermediate object | exact tensor shape and semantic meaning |
| G-1 | whether constructed from raw or contextualized features |
| C-1 | second context equation |
| R-1 | second surviving statistic |
| final head | classification, survival, retrieval, or adaptation |
| S | labels, losses, masks, and split |
| failure tests | order, graph, parent, count, deletion, and readout ablations |
| cost | construction plus all operators |

## 16. Final C/R/G/S table

| Composition | C | R | G | S |
| --- | --- | --- | --- | --- |
| Patch-GCN | coordinate graph context | global attention | physical patch coordinates | survival or slide task |
| MambaMIL | selective scan | attention weighted mean | sequence order | classification or hazard head |
| DTFD | two-tier attention | selection plus parent attention | random pseudo-bags | inherited slide labels |
| HIPT | transformer at nested scales | [CLS] extraction | spatial window hierarchy | DINO plus downstream task |
| HACT | cell and tissue graph context | assignment transfer and coarse readout | cell/tissue graphs and B | source graph classification task |
| generic hybrid | ordered context stack | ordered readout stack | all relation and parent objects | all intermediate and final losses |

## 17. Bottom line

Hybrid MIL is the composition:

```math
\boxed{
\text{object}
\longrightarrow
G_1
\longrightarrow
C_1
\longrightarrow
R_1
\longrightarrow
G_2
\longrightarrow
C_2
\longrightarrow
R_2
\longrightarrow
S\text{-specific head}
}
```

The important mathematical object is not the module list. It is the sequence of
collisions and symmetries induced at every boundary.

A hybrid design is legible when it can answer:

```math
\boxed{
\text{what geometry enters each stage, what context is propagated, what
statistic survives, and which supervision makes the resulting collisions useful?}
}
```
