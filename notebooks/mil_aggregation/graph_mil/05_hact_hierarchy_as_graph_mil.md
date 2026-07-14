# HACT: Cell-to-Tissue Hierarchy as Graph MIL

Sources:

- Pati et al., HACT-Net: A Hierarchical Cell-to-Tissue Graph Neural Network for
  Histopathological Image Classification.
  [arXiv](https://arxiv.org/abs/2007.00584)
- Pati et al., Hierarchical Graph Representations in Digital Pathology.
  [arXiv](https://arxiv.org/abs/2102.11057)
- Published Medical Image Analysis version.
  [DOI](https://doi.org/10.1016/j.media.2021.102264)

HACT is not a flat patch MIL model. It represents a tissue region as a
hierarchical entity graph:

```math
\text{cells}
\longrightarrow
\text{tissue regions}
\longrightarrow
\text{graph-level task representation}.
```

The hierarchy is explicit. It contains:

1. a cell graph for cell morphology and cell-cell relations;
2. a tissue graph for tissue-region morphology and tissue-region relations;
3. a binary assignment matrix connecting every cell to one tissue region.

The defining object is:

```math
\boxed{
\mathcal G_{\mathrm{HACT}}
=
\left(
G_{\mathrm{CG}},
G_{\mathrm{TG}},
B_{\mathrm{CG}\to\mathrm{TG}}
\right)
}
```

This note uses “graph MIL” in the broad representation sense: a set of fine
entities is transformed into one graph-level prediction through context and
readout. HACT was originally proposed for annotated tissue regions of interest
and breast-cancer subtyping, not as a generic whole-slide survival model. The
same operators can be embedded in a larger WSI pipeline, but that extension
should be named as an extension.

## 1. The hierarchical graph object

Let the cell graph be:

```math
G_{\mathrm{CG}}
=
\left(
V_{\mathrm{CG}},
E_{\mathrm{CG}},
H_{\mathrm{CG}}
\right).
```

Let the tissue graph be:

```math
G_{\mathrm{TG}}
=
\left(
V_{\mathrm{TG}},
E_{\mathrm{TG}},
H_{\mathrm{TG}}
\right).
```

The cell node feature matrix is:

```math
H_{\mathrm{CG}}
\in
\mathbb R^{N_{\mathrm C}\times d_{\mathrm C}}.
```

The tissue node feature matrix is:

```math
H_{\mathrm{TG}}
\in
\mathbb R^{N_{\mathrm T}\times d_{\mathrm T}}.
```

The cell-to-tissue assignment matrix is:

```math
B
=
B_{\mathrm{CG}\to\mathrm{TG}}
\in
\{0,1\}^{N_{\mathrm C}\times N_{\mathrm T}}.
```

The full HACT representation is:

```math
\mathcal G_{\mathrm{HACT}}
=
\left(
G_{\mathrm{CG}},
G_{\mathrm{TG}},
B
\right).
```

The matrices for within-level supports are:

```math
A_{\mathrm C}
\in
\{0,1\}^{N_{\mathrm C}\times N_{\mathrm C}},
\qquad
A_{\mathrm T}
\in
\{0,1\}^{N_{\mathrm T}\times N_{\mathrm T}}.
```

The hierarchy is not recoverable from the tissue graph alone after B is
discarded. Two HACT objects can share the same tissue nodes and tissue edges
while differing in which cells belong to each tissue region.

## 2. Cell and tissue entities

The fine nodes are nuclei or cells extracted from the tissue image. A cell
feature can include morphology and appearance:

```math
h_{\mathrm C,a}^{(0)}
=
\psi_{\mathrm C}
\left(
\text{cell }a
\right)
\in
\mathbb R^{d_{\mathrm C}}.
```

A tissue-region feature can include region morphology and appearance:

```math
h_{\mathrm T,b}^{(0)}
=
\psi_{\mathrm T}
\left(
\text{region }b
\right)
\in
\mathbb R^{d_{\mathrm T}}.
```

The two node sets are not merely two random clusters of patch embeddings:

```math
V_{\mathrm C}
=
\text{fine histological entities},
```

```math
V_{\mathrm T}
=
\text{coarse tissue entities}.
```

The graph supports represent distinct intra-level relations:

```math
E_{\mathrm C}
=
\text{cell-cell relations},
\qquad
E_{\mathrm T}
=
\text{tissue-region relations}.
```

The hierarchy represents relative spatial distribution of cells with respect to
tissue regions.

## 3. Assignment construction

Let c-a be the centroid of cell a and R-b the region mask for tissue node b. A
hard assignment is:

```math
B_{ab}
=
\mathbf 1
\left\{
c_a\in R_b
\right\}.
```

The paper states that every nucleus is assigned to one tissue region. Thus:

```math
\sum_{b=1}^{N_{\mathrm T}}
B_{ab}
=
1
\qquad
\text{for every cell }a.
```

If a segmented nucleus overlaps multiple tissue regions, assign it to the
region with maximum overlap:

```math
b^\star(a)
=
\arg\max_b
\mathrm{Area}
\left(
\mathrm{Nucleus}_a
\cap
R_b
\right),
```

```math
B_{ab}
=
\mathbf 1\{b=b^\star(a)\}.
```

The columns need not sum to one. A tissue region can contain many cells:

```math
n_b^{\mathrm C}
=
\sum_{a=1}^{N_{\mathrm C}}B_{ab}.
```

The assignment matrix is therefore row-stochastic in the hard one-parent sense,
not column-stochastic.

## 4. Assignment as a geometry operator

The map B encodes more than a pooling partition. It identifies a geometric
parent relation:

```math
a\to b
\quad
\Longleftrightarrow
\quad
B_{ab}=1.
```

The relation is determined by cell centroid and tissue-region geometry. It is
not a random pseudo-bag split.

Compare:

```math
B_{\mathrm{HACT}}
=
\Gamma_{\mathrm{cell\text{-}region}}
\left(
\text{masks and centroids}
\right),
```

with DTFD pseudo-bags:

```math
B_{\mathrm{DTFD}}
=
\Gamma_{\mathrm{random}}
\left(
\text{patch indices}
\right).
```

Both create child-to-parent groups, but only the first is an explicit anatomical
or semantic hierarchy.

## 5. Cell-graph context

Let the cell graph context operator be:

```math
\widetilde H_{\mathrm C}
=
\mathcal C_{\mathrm C}
\left(
H_{\mathrm C}^{(0)},A_{\mathrm C}
\right)
\in
\mathbb R^{N_{\mathrm C}\times d_{\mathrm C}'}.
```

A generic message-passing update is:

```math
m_{\mathrm C,a}^{(\ell+1)}
=
\rho_{\mathrm C}^{(\ell)}
\left(
\left\{
\phi_{\mathrm C}^{(\ell)}
\left(
h_{\mathrm C,a}^{(\ell)},
h_{\mathrm C,u}^{(\ell)}
\right)
:
u\in\mathcal N_{\mathrm C}(a)
\right\}
\right),
```

```math
h_{\mathrm C,a}^{(\ell+1)}
=
\zeta_{\mathrm C}^{(\ell)}
\left(
h_{\mathrm C,a}^{(\ell)},
m_{\mathrm C,a}^{(\ell+1)}
\right).
```

The original HACT-Net paper permits a GNN architecture for CG-GNN. The later
work instantiates HACT-Net using PNA layers. The exact choice of cell context
is therefore version-dependent.

## 6. GIN-style cell update in HACT-Net

A dimension-safe reconstruction of the original self-plus-neighbor update is:

```math
h_{\mathrm C,a}^{(\ell+1)}
=
\mathrm{MLP}_{\ell}
\left(
(1+\epsilon_\ell)h_{\mathrm C,a}^{(\ell)}
+
\sum_{u\in\mathcal N_{\mathrm C}(a)}
h_{\mathrm C,u}^{(\ell)}
\right).
```

The simpler self-plus-neighbor form is:

```math
h_{\mathrm C,a}^{(\ell+1)}
=
\mathrm{MLP}_{\ell}
\left(
h_{\mathrm C,a}^{(\ell)}
+
\sum_{u\in\mathcal N_{\mathrm C}(a)}
h_{\mathrm C,u}^{(\ell)}
\right).
```

The learnable self coefficient epsilon should only be included when the
implementation uses it. Both formulas share the relevant C/R/G distinction:
the cell context is an invariant neighbor multiset aggregation followed by a
shared node update.

## 7. PNA cell context in the later formulation

The later HACT-Net work uses Principal Neighbourhood Aggregation. For cell a
at layer t, first compute messages:

```math
m_{\mathrm C,a,u}^{(t)}
=
M_{\mathrm C}^{(t)}
\left(
h_{\mathrm C,a}^{(t)},
h_{\mathrm C,u}^{(t)}
\right),
\qquad
u\in\mathcal N_{\mathrm C}(a).
```

Let the neighbor multiset be:

```math
D_{\mathrm C,a}^{(t)}
=
\left\{
m_{\mathrm C,a,u}^{(t)}
:
u\in\mathcal N_{\mathrm C}(a)
\right\}.
```

PNA applies multiple aggregators:

```math
\mathrm{Agg}
\left(
D_{\mathrm C,a}^{(t)}
\right)
=
\left[
\mu(D_{\mathrm C,a}^{(t)}),
\sigma(D_{\mathrm C,a}^{(t)}),
\min(D_{\mathrm C,a}^{(t)}),
\max(D_{\mathrm C,a}^{(t)})
\right].
```

For a vector-valued message, mean and standard deviation are coordinatewise:

```math
\mu_r(D)
=
\frac{1}{|D|}
\sum_{x\in D}x_r,
```

```math
\sigma_r(D)
=
\sqrt{
\frac{1}{|D|}
\sum_{x\in D}
\left(
x_r-\mu_r(D)
\right)^2
}.
```

Degree scalers adjust the aggregated statistics. Let d-a be node degree and
delta be the training-set average of log degree plus one. A generic scaler is:

```math
S(d_a;\alpha)
=
\left(
\frac{\log(d_a+1)}{\delta}
\right)^\alpha,
\qquad
\alpha\in\{-1,0,1\}.
```

The PNA tensor product combines scalers and aggregators:

```math
\mathrm{PNA}_{a}^{(t)}
=
\bigotimes_{\alpha\in\{-1,0,1\}}
\left[
S(d_a;\alpha)
\mathrm{Agg}
\left(
D_{\mathrm C,a}^{(t)}
\right)
\right].
```

The node update is:

```math
h_{\mathrm C,a}^{(t+1)}
=
U_{\mathrm C}^{(t)}
\left(
h_{\mathrm C,a}^{(t)},
\mathrm{PNA}_{a}^{(t)}
\right).
```

PNA retains more neighborhood statistics than a single mean. It is still a
permutation-invariant local aggregator; it does not retain the ordered list of
neighbors.

## 8. Jumping knowledge

After T-CG cell layers, the later formulation uses an LSTM-based jumping
knowledge operator:

```math
h_{\mathrm C,a}^{(T_{\mathrm C}+1)}
=
\mathrm{LSTM}
\left(
h_{\mathrm C,a}^{(1)},
\ldots,
h_{\mathrm C,a}^{(T_{\mathrm C})}
\right).
```

This retains a learned combination of shallow and deep cell-context states.
The final cell representation is not merely the output of the deepest PNA
layer.

The surviving cell statistic is multiscale:

```math
h_{\mathrm C,a}^{\mathrm{JK}}
=
\Psi_{\mathrm{JK}}
\left(
h_{\mathrm C,a}^{(1:T_{\mathrm C})}
\right).
```

This matters at the hierarchy boundary. The coarse tissue node receives a
summary of the cell's context at multiple depths.

## 9. Cell-to-tissue transfer

Let:

```math
\widetilde H_{\mathrm C}
=
\begin{bmatrix}
\widetilde h_{\mathrm C,1}^{\mathsf T}\\
\vdots\\
\widetilde h_{\mathrm C,N_{\mathrm C}}^{\mathsf T}
\end{bmatrix}
\in
\mathbb R^{N_{\mathrm C}\times d_{\mathrm C}'}.
```

The assignment sum is:

```math
U_{\mathrm C\to T}
=
B^{\mathsf T}\widetilde H_{\mathrm C}
\in
\mathbb R^{N_{\mathrm T}\times d_{\mathrm C}'}.
```

For tissue region b:

```math
u_{\mathrm T,b}
=
\sum_{a:B_{ab}=1}
\widetilde h_{\mathrm C,a}.
```

The matrix product is exact because:

```math
\left[
B^{\mathsf T}\widetilde H_{\mathrm C}
\right]_b
=
\sum_{a=1}^{N_{\mathrm C}}
B_{ab}\widetilde h_{\mathrm C,a}.
```

The later HACT formulation uses a concatenation of original tissue features and
the transferred cell summary:

```math
h_{\mathrm T,b}^{(0)}
=
\left[
H_{\mathrm T,b}
\middle\Vert
u_{\mathrm T,b}
\right].
```

If the feature widths are d-T and d-C-prime, then:

```math
h_{\mathrm T,b}^{(0)}
\in
\mathbb R^{d_{\mathrm T}+d_{\mathrm C}'}.
```

The cell-to-tissue map is therefore both a readout and a context transfer
operator:

```math
\mathcal R_{\mathrm C\to T}
=
B^{\mathsf T},
\qquad
\mathcal C_{\mathrm T}^{(0)}
=
\mathrm{Concat}
\left(
H_{\mathrm T},
B^{\mathsf T}\widetilde H_{\mathrm C}
\right).
```

## 10. Sum versus mean transfer

HACT's transfer uses a sum over assigned cell representations. A mean transfer
would be:

```math
\overline u_{\mathrm T,b}
=
\frac{
1
}{
n_b^{\mathrm C}
}
\sum_{a:B_{ab}=1}
\widetilde h_{\mathrm C,a}.
```

The two have different cardinality contracts:

```math
u_{\mathrm T,b}^{\mathrm{sum}}
=
n_b^{\mathrm C}
\overline u_{\mathrm T,b}.
```

Sum preserves cellularity and feature mass. Mean is invariant to equal
duplication of cells within each tissue region. Neither is universally correct.

If the tissue label depends on density or cellular burden, sum can retain useful
signal. If cell count reflects segmentation quality or region area, sum can
encode nuisance exposure.

The paper-specific derivation should preserve the sum transfer where it is
used. A mean variant is a controlled alternative, not an algebraic synonym.

## 11. Tissue-graph context

Let:

```math
H_{\mathrm T}^{(0)}
=
\left[
H_{\mathrm T}
\middle\Vert
B^{\mathsf T}\widetilde H_{\mathrm C}
\right].
```

The tissue graph context operator is:

```math
\widetilde H_{\mathrm T}
=
\mathcal C_{\mathrm T}
\left(
H_{\mathrm T}^{(0)},A_{\mathrm T}
\right).
```

A generic update is:

```math
m_{\mathrm T,b}^{(\ell+1)}
=
\rho_{\mathrm T}^{(\ell)}
\left(
\left\{
\phi_{\mathrm T}^{(\ell)}
\left(
h_{\mathrm T,b}^{(\ell)},
h_{\mathrm T,r}^{(\ell)}
\right)
:
r\in\mathcal N_{\mathrm T}(b)
\right\}
\right),
```

```math
h_{\mathrm T,b}^{(\ell+1)}
=
\zeta_{\mathrm T}^{(\ell)}
\left(
h_{\mathrm T,b}^{(\ell)},
m_{\mathrm T,b}^{(\ell+1)}
\right).
```

The tissue graph adds cross-region context after the cell summaries have been
transferred.

The full context order is:

```math
H_{\mathrm C}
\longrightarrow
\mathcal C_{\mathrm C}
\longrightarrow
B^{\mathsf T}\widetilde H_{\mathrm C}
\longrightarrow
\mathcal C_{\mathrm T}.
```

Replacing this with tissue context first and cell context second gives a
different model.

## 12. PNA tissue context

The later formulation applies PNA to tissue nodes as well. For tissue b:

```math
D_{\mathrm T,b}^{(t)}
=
\left\{
M_{\mathrm T}^{(t)}
\left(
h_{\mathrm T,b}^{(t)},
h_{\mathrm T,r}^{(t)}
\right)
:
r\in\mathcal N_{\mathrm T}(b)
\right\}.
```

The same family of aggregators and degree scalers can be applied:

```math
\mathrm{PNA}_{\mathrm T,b}^{(t)}
=
\bigotimes_{\alpha\in\{-1,0,1\}}
\left[
S(d_{\mathrm T,b};\alpha)
\mathrm{Agg}
\left(
D_{\mathrm T,b}^{(t)}
\right)
\right].
```

The update is:

```math
h_{\mathrm T,b}^{(t+1)}
=
U_{\mathrm T}^{(t)}
\left(
h_{\mathrm T,b}^{(t)},
\mathrm{PNA}_{\mathrm T,b}^{(t)}
\right).
```

This creates two different degree distributions:

```math
d_{\mathrm C,a}
=
|\mathcal N_{\mathrm C}(a)|,
\qquad
d_{\mathrm T,b}
=
|\mathcal N_{\mathrm T}(b)|.
```

The degree scalers should be interpreted at the level where they are applied.
A cell degree and a tissue degree are not interchangeable statistics.

## 13. Graph-level readout

Let the final tissue states be:

```math
H_{\mathrm T}^{\star}
=
\left\{
h_{\mathrm T,b}^{\star}
\right\}_{b=1}^{N_{\mathrm T}}.
```

A tissue-level sum readout is:

```math
z_{\mathrm{sum}}
=
\sum_{b=1}^{N_{\mathrm T}}
h_{\mathrm T,b}^{\star}.
```

A mean readout is:

```math
z_{\mathrm{mean}}
=
\frac{1}{N_{\mathrm T}}
\sum_{b=1}^{N_{\mathrm T}}
h_{\mathrm T,b}^{\star}.
```

The short HACT-Net formulation describes layerwise sum-concatenation:

```math
z_{\mathrm{JK\text{-}sum}}
=
\mathop{\mathrm{Concat}}_{\ell}
\left(
\sum_{b=1}^{N_{\mathrm T}}
h_{\mathrm T,b}^{(\ell)}
\right).
```

A graph representation is then classified by an MLP:

```math
\widehat y
=
\mathrm{softmax}
\left(
W_{\mathrm{cls}}z+b_{\mathrm{cls}}
\right).
```

The graph-level surviving statistic is a tissue-node statistic after two
within-level context operators and one fine-to-coarse transfer.

## 14. HACT-Net forward map

The complete later-version map is:

```math
H_{\mathrm C}^{(0)}
\xrightarrow{
\mathcal C_{\mathrm C}^{\mathrm{PNA+JK}}
}
\widetilde H_{\mathrm C}
\xrightarrow{
B^{\mathsf T}\text{ sum transfer}
}
U_{\mathrm C\to T}
\xrightarrow{
\mathrm{Concat}(H_{\mathrm T},\cdot)
}
H_{\mathrm T}^{(0)}
\xrightarrow{
\mathcal C_{\mathrm T}^{\mathrm{PNA}}
}
\widetilde H_{\mathrm T}
\xrightarrow{
\mathcal R_{\mathrm T}
}
z
\xrightarrow{
\mathcal H
}
\widehat y.
```

The cell graph and tissue graph are both context operators. B is a geometry
and transfer operator. The final tissue readout is the graph-level R.

## 15. Assignment nullspace

The linear transfer is:

```math
\Pi_B(H)
=
B^{\mathsf T}H.
```

Let Delta be a cell-state perturbation satisfying:

```math
B^{\mathsf T}\Delta
=
0.
```

Then:

```math
\Pi_B(H+\Delta)
=
B^{\mathsf T}(H+\Delta)
=
B^{\mathsf T}H.
```

The coarse initialization is unchanged:

```math
H_{\mathrm T}^{(0)}(H+\Delta)
=
H_{\mathrm T}^{(0)}(H).
```

Every zero-sum redistribution of contextualized cell states within each tissue
region lies in the nullspace of the assignment sum.

For a hard assignment with every tissue region nonempty and rank B equal to
N-T, the scalar-feature nullity is:

```math
\mathrm{nullity}(B^{\mathsf T})
=
N_{\mathrm C}-N_{\mathrm T}.
```

For d-C-prime feature coordinates, the transfer discards up to:

```math
\left(
N_{\mathrm C}-N_{\mathrm T}
\right)d_{\mathrm C}'
```

degrees of freedom before tissue context, subject to the rank and
independence assumptions.

This is not an error. It is the intended coarse representation bottleneck.

## 16. Within-region collision

Suppose two cell-state fields differ by Delta:

```math
\widetilde H_{\mathrm C}'
=
\widetilde H_{\mathrm C}
+
\Delta,
```

with:

```math
\sum_{a:B_{ab}=1}
\Delta_a
=
0
\quad
\text{for every tissue region }b.
```

Then:

```math
B^{\mathsf T}\widetilde H_{\mathrm C}'
=
B^{\mathsf T}\widetilde H_{\mathrm C}.
```

The tissue graph receives the same transferred features. If original tissue
features and tissue adjacency also match, the entire tissue-level predictor
collides:

```math
\widehat y(\widetilde H_{\mathrm C}')
=
\widehat y(\widetilde H_{\mathrm C}).
```

HACT cannot distinguish within-region configurations whose contextualized
cell sums agree unless additional within-region statistics are transferred.

## 17. Why cell context before transfer matters

If the assignment is applied before cell context:

```math
\mathcal C_{\mathrm C}
\left(
B^{\mathsf T}H_{\mathrm C}
\right),
```

the cell graph no longer has access to individual cell states. This is not the
HACT sequence.

HACT applies:

```math
B^{\mathsf T}
\mathcal C_{\mathrm C}(H_{\mathrm C},A_{\mathrm C}),
```

which allows a cell to exchange information with its cell neighbors before its
state is summed into a tissue region.

In general:

```math
B^{\mathsf T}
\mathcal C_{\mathrm C}(H_{\mathrm C},A_{\mathrm C})
\ne
\mathcal C_{\mathrm T}
\left(
B^{\mathsf T}H_{\mathrm C},
A_{\mathrm T}
\right).
```

The two sides have different domains and preserve different fine-scale
information.

## 18. Assignment and permutation equivariance

Let P-C permute cells and P-T permute tissue regions. Transform:

```math
H_{\mathrm C}'
=
P_{\mathrm C}H_{\mathrm C},
\qquad
H_{\mathrm T}'
=
P_{\mathrm T}H_{\mathrm T}.
```

The assignment transforms as:

```math
B'
=
P_{\mathrm C}BP_{\mathrm T}^{\mathsf T}.
```

Then:

```math
B'^{\mathsf T}H_{\mathrm C}'
=
P_{\mathrm T}B^{\mathsf T}P_{\mathrm C}^{\mathsf T}
P_{\mathrm C}H_{\mathrm C}
=
P_{\mathrm T}B^{\mathsf T}H_{\mathrm C}.
```

The transferred tissue rows are reindexed by P-T. If cell and tissue graph
operators are equivariant and the final readout is invariant, the graph
prediction is invariant to storage order at both levels.

The semantic parent relation is preserved under coordinated permutations.

## 19. Assignment errors

Let B-star be the correct assignment and B-hat the constructed assignment. The
assignment error is:

```math
E_B
=
\widehat B-B^\star.
```

The transferred feature error is:

```math
\Delta U
=
\widehat B^{\mathsf T}\widetilde H_{\mathrm C}
-
B^{\star\mathsf T}\widetilde H_{\mathrm C}
=
E_B^{\mathsf T}\widetilde H_{\mathrm C}.
```

A cell assigned to the wrong tissue region moves its entire contextualized
feature contribution between coarse nodes.

The error is not necessarily averaged away by the tissue graph. It can change:

```text
tissue-node initialization
tissue degrees if graph construction uses assignments
coarse message paths
final graph readout
```

A model can therefore learn assignment artifacts if segmentation or region
definitions correlate with the task.

## 20. Empty and large tissue regions

For tissue region b, the number of assigned cells is:

```math
n_b^{\mathrm C}
=
\sum_aB_{ab}.
```

If n-b-C equals zero, the transferred sum is zero:

```math
u_{\mathrm T,b}
=
0.
```

If one region contains a very large fraction of cells, its sum has a larger norm
under comparable cell features:

```math
\left\|
u_{\mathrm T,b}
\right\|
\le
n_b^{\mathrm C}
\max_{a:B_{ab}=1}
\|\widetilde h_{\mathrm C,a}\|.
```

This creates a count or region-area pathway. A mean transfer would remove that
linear count dependence but introduce a different estimator.

The implementation should state how empty regions are represented and whether
tissue-region sizes are available to the final readout.

## 21. Cell graph and tissue graph are not redundant

A cell-only graph retains:

```math
\mathcal C_{\mathrm C}
\left(
H_{\mathrm C},A_{\mathrm C}
\right)
\longrightarrow
\mathcal R_{\mathrm C}.
```

A tissue-only graph retains:

```math
\mathcal C_{\mathrm T}
\left(
H_{\mathrm T},A_{\mathrm T}
\right)
\longrightarrow
\mathcal R_{\mathrm T}.
```

HACT retains both plus the transfer:

```math
\mathcal C_{\mathrm T}
\left(
\left[
H_{\mathrm T}
\middle\Vert
B^{\mathsf T}\mathcal C_{\mathrm C}(H_{\mathrm C},A_{\mathrm C})
\right],
A_{\mathrm T}
\right).
```

The ablation comparison tests whether cell context, tissue context, and
inter-level transfer each contribute. It is not enough to compare graph layer
counts while changing the entity graph.

## 22. HACT versus flat graph MIL

Flat graph MIL:

```math
z_{\mathrm{flat}}
=
\mathcal R
\left(
\mathcal C(H,A)
\right).
```

HACT:

```math
z_{\mathrm{HACT}}
=
\mathcal R_{\mathrm T}
\left(
\mathcal C_{\mathrm T}
\left(
H_{\mathrm T}
\Vert
B^{\mathsf T}
\mathcal C_{\mathrm C}(H_{\mathrm C},A_{\mathrm C}),
A_{\mathrm T}
\right)
\right).
```

The flat graph has one node scale. HACT has two node scales and an explicit
assignment map.

A flat graph can emulate some hierarchical functions if it receives enough
features and relations, but it does not automatically expose the same
fine-to-coarse coordinate system.

## 23. HACT versus DTFD

DTFD pseudo-bag partition:

```math
\{1,\ldots,N\}
=
\biguplus_{m=1}^{M}\mathcal B_m.
```

HACT assignment:

```math
B_{ab}
=
\mathbf 1
\left\{
\text{cell }a\text{ belongs to tissue region }b
\right\}.
```

DTFD inherits the slide label:

```math
\widetilde Y_m=Y.
```

HACT's cell and tissue nodes are representation entities; the slide or region
label is applied to the graph-level output. The assignment itself is not a
copied weak label.

The two models share the composition pattern:

```math
\text{fine states}
\longrightarrow
\text{coarse states}
\longrightarrow
\text{global prediction},
```

but differ in geometry and supervision.

## 24. HACT versus HIPT

HIPT uses nested image windows and transformer tokens. HACT uses explicit graph
entities and assignment B.

HIPT parent relation:

```math
\mathcal P_{\mathrm{HIPT}}
=
\text{spatial window containment}.
```

HACT parent relation:

```math
\mathcal P_{\mathrm{HACT}}
=
\text{cell-to-tissue assignment}.
```

HIPT context is token self-attention. HACT context is message passing on cell
and tissue graphs.

Both can lose information at a coarse token boundary. HACT's linear assignment
nullspace makes one such loss explicit.

## 25. Graph-level task head

For C classes:

```math
\widehat p
=
\mathrm{softmax}
\left(
W_{\mathrm{cls}}z+b_{\mathrm{cls}}
\right).
```

For one-hot target y:

```math
\mathcal L_{\mathrm{CE}}
=
-
\sum_{c=1}^{C}
y_c\log(\widehat p_c).
```

The HACT papers focus on histopathological image or tissue-region
classification. A WSI-level use would require a defined mechanism for
constructing cell and tissue graphs across the slide and a slide-level target.
It should not be described as the original task without that extension.

For a survival extension, one could define:

```math
\eta_i
=
w^{\mathsf T}z_i+b,
```

and use a survival loss, but this is a new task head:

```math
\mathcal L_{\mathrm{surv}}
=
\mathcal L_{\mathrm{Cox}}
\left(
\{\eta_i,T_i,\delta_i\}_i
\right).
```

The graph representation and task head must be separated.

## 26. C/R/G/S placement

| Component | HACT instantiation |
| --- | --- |
| C | cell-graph message passing, cell-context jumping knowledge, cell-to-tissue transfer, tissue-graph message passing |
| R | assignment sum or concatenated fine summaries, followed by tissue-node sum or layerwise sum-concatenation and an MLP head |
| G | cell graph, tissue graph, and binary cell-to-tissue assignment matrix based on spatial entity containment |
| S | region or slide classification labels; cell/tissue structures and assignments are upstream representation objects |

The forward map is:

```math
\left(
H_{\mathrm C},A_{\mathrm C},H_{\mathrm T},A_{\mathrm T},B
\right)
\longrightarrow
\widetilde H_{\mathrm C}
\longrightarrow
B^{\mathsf T}\widetilde H_{\mathrm C}
\longrightarrow
H_{\mathrm T}^{(0)}
\longrightarrow
\widetilde H_{\mathrm T}
\longrightarrow
z
\longrightarrow
\widehat y.
```

The four axes should not be compressed into “hierarchical graph pooling”:

```math
\mathcal C
\ne
\mathcal R
\ne
\mathcal G
\ne
\mathcal S.
```

## 27. Sanity checks

### 27.1 Assignment row sum

Verify one parent per cell:

```math
B\mathbf 1_{N_{\mathrm T}}
=
\mathbf 1_{N_{\mathrm C}}.
```

A violation means the hierarchy is not the hard one-parent object assumed by
the derivation.

### 27.2 Assignment transfer shape

Check:

```math
B^{\mathsf T}
\in
\mathbb R^{N_{\mathrm T}\times N_{\mathrm C}},
\qquad
H_{\mathrm C}
\in
\mathbb R^{N_{\mathrm C}\times d_{\mathrm C}},
```

so:

```math
B^{\mathsf T}H_{\mathrm C}
\in
\mathbb R^{N_{\mathrm T}\times d_{\mathrm C}}.
```

### 27.3 Nullspace collision

Construct Delta with zero sum inside every tissue region:

```math
B^{\mathsf T}\Delta=0.
```

Verify the transferred features match:

```math
B^{\mathsf T}(H_{\mathrm C}+\Delta)
=
B^{\mathsf T}H_{\mathrm C}.
```

### 27.4 Cell-context order

Compare:

```math
B^{\mathsf T}
\mathcal C_{\mathrm C}(H_{\mathrm C},A_{\mathrm C})
```

with:

```math
\mathcal C_{\mathrm T}
\left(
B^{\mathsf T}H_{\mathrm C},
A_{\mathrm T}
\right).
```

They should not be treated as the same operation.

### 27.5 Count sensitivity

Duplicate one cell within a tissue region and compare sum versus mean transfer.
The sum should change linearly with the duplicate; the mean should not.

### 27.6 Graph-level permutation

Permute cells and tissue nodes with coordinated B transformation. The final
graph prediction should be invariant.

### 27.7 Assignment perturbation

Move one cell to a neighboring tissue region and recompute:

```math
\Delta z
=
z(B')-z(B).
```

This tests sensitivity to parent-map errors.

## 28. Bottom line

HACT is an explicit hierarchical graph representation:

```math
\boxed{
\mathrm{HACT}
=
\left(
\mathrm{cell\ graph},
\mathrm{tissue\ graph},
\mathrm{cell\text{-}to\text{-}tissue\ assignment}
\right)
}
```

The later HACT-Net map is:

```math
\boxed{
z
=
\mathcal R_{\mathrm T}
\circ
\mathcal C_{\mathrm T}
\left(
H_{\mathrm T}
\Vert
B^{\mathsf T}
\mathcal C_{\mathrm C}(H_{\mathrm C},A_{\mathrm C}),
A_{\mathrm T}
\right)
}
```

The assignment transfer is a real information bottleneck:

```math
B^{\mathsf T}\Delta=0
\Longrightarrow
B^{\mathsf T}(H_{\mathrm C}+\Delta)
=
B^{\mathsf T}H_{\mathrm C}.
```

The surviving statistic is a coarse tissue representation built from
contextualized cell sums, followed by tissue-level context and readout.

The C/R/G/S summary is:

```math
\boxed{
\text{fine graph context}
+
\text{assignment-sum transfer}
+
\text{coarse graph context}
+
\text{graph-level task readout}
}
```

The right question is not whether HACT has “two GNNs.” It is:

```math
\text{which within-tissue cellular configurations survive the assignment
bottleneck and remain available to the tissue graph?}
```
