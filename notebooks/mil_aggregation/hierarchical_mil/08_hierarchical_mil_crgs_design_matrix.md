# Hierarchical MIL C/R/G/S Design Matrix

Hierarchical MIL is a composition of boundary operators, not merely a model with
multiple resolutions. For level ell, define:

```math
H^{(\ell+1)}
=
\mathcal R_{\ell,\theta}
\left(
\mathcal C_{\ell,\theta}
\left(
H^{(\ell)},G^{(\ell)}
\right),
P_\ell
\right).
```

The slide prediction is:

```math
z_i
=
\mathcal R_{\mathrm{slide},\theta}
\left(
H_i^{(L)},G_i^{(L)}
\right),
\qquad
\widehat y_i
=
\mathcal H_\omega(z_i).
```

The four axes are:

```math
\boxed{
\begin{aligned}
G &: \text{parent maps, scale geometry, and support relations},
\\
C &: \text{within-level and cross-level context},
\\
R &: \text{child-to-parent and parent-to-slide statistics},
\\
S &: \text{labels and losses acting at each level}.
\end{aligned}
}
```

A hierarchy is mathematically legible only when every boundary is specified.

## 1. The hierarchical object

At level ell, let:

```math
V^{(\ell)}
=
\{1,\ldots,N_\ell\},
\qquad
H^{(\ell)}
\in
\mathbb R^{N_\ell\times d_\ell}.
```

A parent map sends children to parents:

```math
P_\ell:
V^{(\ell)}
\longrightarrow
V^{(\ell+1)}.
```

For a hard assignment matrix:

```math
B_\ell
\in
\{0,1\}^{N_\ell\times N_{\ell+1}},
\qquad
\sum_{b=1}^{N_{\ell+1}}
[B_\ell]_{ab}
=
1.
```

The complete hierarchy object is:

```math
\mathcal H_i
=
\left(
\{H_i^{(\ell)}\}_{\ell=0}^{L},
\{G_i^{(\ell)}\}_{\ell=0}^{L},
\{B_{i,\ell}\}_{\ell=0}^{L-1}
\right).
```

The parent map can be:

```text
random:
    DTFD pseudo-bag membership

spatial:
    HIPT nested fixed windows

graph-geometric:
    HACT cell-to-tissue assignment

learned:
    routing probabilities or differentiable clustering
```

These choices define different G operators even if the same attention block is
used at every level.

## 2. Four-axis decomposition

### Geometry G

G includes the parent map and any within-level relation:

```math
G^{(\ell)}
=
\Gamma_{\mathrm G}^{(\ell)}
\left(
H^{(\ell)},P^{(\ell)},\tau^{(\ell)},\phi^{(\ell)}
\right).
```

### Context C

C acts before a boundary:

```math
\widetilde H^{(\ell)}
=
\mathcal C_{\ell,\theta}
\left(
H^{(\ell)},G^{(\ell)}
\right).
```

It can also act after coarsening:

```math
H^{(\ell+1)}
=
\mathcal C_{\ell+1,\theta}
\left(
\mathcal R_{\ell,\theta}
(\widetilde H^{(\ell)},B_\ell),
G^{(\ell+1)}
\right).
```

### Readout R

R is every child-to-parent or parent-to-slide compression:

```math
H^{(\ell+1)}
=
\mathcal R_{\ell,\theta}
\left(
\widetilde H^{(\ell)},B_\ell
\right).
```

It can be sum, mean, attention, selection, [CLS], prototype, or a learned
routing statistic.

### Supervision S

S includes all labels and objectives:

```math
S
=
\left(
\mathcal Y_{\mathrm{slide}},
\mathcal Y_{\mathrm{level}},
\mathcal L_{\mathrm{task}},
\mathcal L_{\mathrm{aux}},
\mathcal D_{\mathrm{split}}
\right).
```

Inherited pseudo-bag labels, DINO views, cell labels, and survival outcomes are
different supervision channels even when they influence the same parameters.

## 3. General forward map

For one level with assignment B:

```math
\widetilde H
=
\mathcal C_\theta(H,G),
```

```math
H_{\mathrm{parent}}
=
\mathcal R_\theta(\widetilde H,B).
```

For a linear assignment sum:

```math
H_{\mathrm{parent}}
=
B^{\mathsf T}\widetilde H.
```

For a parent mean:

```math
[H_{\mathrm{parent}}]_{b,:}
=
\frac{
\sum_aB_{ab}\widetilde h_a
}{
\sum_aB_{ab}
}.
```

For a within-parent attention readout:

```math
\alpha_{ab}
=
\frac{
B_{ab}\exp(s_{ab})
}{
\sum_cB_{cb}\exp(s_{cb})
},
```

```math
[H_{\mathrm{parent}}]_{b,:}
=
\sum_a\alpha_{ab}\widetilde h_a.
```

The parent-level context then acts on the resulting field:

```math
\widetilde H_{\mathrm{parent}}
=
\mathcal C_{\mathrm{parent}}
\left(
H_{\mathrm{parent}},G_{\mathrm{parent}}
\right).
```

The order is important:

```math
\mathcal R_{\mathrm{parent}}
\circ
\mathcal C_{\mathrm{child}}
\ne
\mathcal C_{\mathrm{parent}}
\circ
\mathcal R_{\mathrm{child}}
```

in general.

## 4. Paper-specific matrix

| Method | G: hierarchy object | C: context | R: boundary statistic | S: supervision |
| --- | --- | --- | --- | --- |
| DTFD-MIL | random balanced pseudo-bags | Tier-1 and Tier-2 gated attention | MaxS, MaxMinS, MAS, or AFS, then Tier-2 attention | inherited slide labels at Tier 1 and slide loss at Tier 2 |
| HIPT | nested spatial windows | transformer self-attention and FFN at each scale | [CLS] extraction at cell, region, and slide scales | DINO pretraining plus downstream slide task |
| HACT | cell graph, tissue graph, hard assignment | cell graph context, transfer, tissue graph context | assignment transfer then tissue readout | source task uses region-level graph hierarchy and slide labels |
| generic hierarchical MIL | fixed, random, spatial, graph, or learned parent map | arbitrary level-specific context | sum, mean, attention, selection, prototype, or [CLS] | level-specific and slide-level objectives |

DTFD's pseudo-bags are not anatomical regions. HIPT's windows are spatial
computational units, not guaranteed biological compartments. HACT's parent map
is an explicit cell-to-tissue relation.

## 5. DTFD forward map

Let the flat patch bag be:

```math
B_i
=
\{h_{ij}\}_{j=1}^{n_i}.
```

Randomly partition it:

```math
B_i
=
\biguplus_{r=1}^{R}B_{ir}.
```

Tier-1 attention is:

```math
a_{irj}
=
\frac{\exp(s_\theta(h_{irj}))}
{\sum_{u\in B_{ir}}\exp(s_\theta(h_{iru}))},
```

```math
z_{ir}^{(1)}
=
\sum_{j\in B_{ir}}
a_{irj}h_{irj}.
```

The operational class-weighted instance signal is:

```math
g_{irj,c}
=
a_{irj}
w_{1,c}^{\mathsf T}h_{irj}.
```

Selection produces d-ir:

```math
d_{ir}
=
\mathcal R_{\mathrm{select}}
\left(
\{h_{irj},g_{irj,c},a_{irj}\}_{j\in B_{ir}}
\right).
```

Tier 2 receives:

```math
D_i
=
\{d_{ir}\}_{r=1}^{R}.
```

Tier-2 attention is:

```math
\beta_{ir}
=
\frac{\exp(t_\psi(d_{ir}))}
{\sum_s\exp(t_\psi(d_{is}))},
```

```math
z_i^{(2)}
=
\sum_r\beta_{ir}d_{ir}.
```

The composition is:

```math
F_{\mathrm{DTFD}}
=
\mathcal H
\circ
\mathcal R_{\mathrm{Tier2}}
\circ
\mathcal R_{\mathrm{select}}
\circ
\mathcal C_{\mathrm{Tier1}}
\circ
\Gamma_{\mathrm{random}}.
```

The random partition is G, selection is R, and inherited labels are S. The
pseudo-bag is not a new observed unit.

## 6. HIPT forward map

At the lowest scale, a 256 by 256 window contains 256 16-by-16 image tokens.
Let:

```math
X_{irp}
=
\{x_{irpc}\}_{c=1}^{256}.
```

The first transformer produces a [CLS] token:

```math
h_{irp}
=
\mathrm{ViT}_{16}^{(1)}
(X_{irp})_{\mathrm{CLS}}.
```

A 4096 by 4096 region contains 256 lower-scale tokens:

```math
H_{ir}
=
\{h_{irp}\}_{p=1}^{256}.
```

The second transformer gives:

```math
h_{ir}
=
\mathrm{ViT}_{256}^{(2)}
(H_{ir})_{\mathrm{CLS}}.
```

The slide contains R-i region tokens:

```math
Z_i
=
\{h_{ir}\}_{r=1}^{R_i}.
```

The third transformer gives:

```math
z_i
=
\mathrm{ViT}_{4096}^{(3)}
(Z_i)_{\mathrm{CLS}}.
```

The composition is:

```math
F_{\mathrm{HIPT}}
=
\mathcal H
\circ
\mathrm{CLS}_3
\circ
\mathcal C_3
\circ
\mathrm{CLS}_2
\circ
\mathcal C_2
\circ
\mathrm{CLS}_1
\circ
\mathcal C_1.
```

The nested spatial windows are G. Self-attention is C. [CLS] extraction is R.
DINO lower-scale objectives and downstream task labels are S.

## 7. HACT forward map

Let cell and tissue features be H-C and H-T, with graph supports A-C and A-T:

```math
\widetilde H_{\mathrm C}
=
\mathcal C_{\mathrm C}
\left(
H_{\mathrm C},A_{\mathrm C}
\right).
```

The binary assignment is:

```math
B
\in
\{0,1\}^{N_{\mathrm C}\times N_{\mathrm T}},
\qquad
B\mathbf 1_{N_{\mathrm T}}
=
\mathbf 1_{N_{\mathrm C}}.
```

Transfer is:

```math
U_{\mathrm C\to T}
=
B^{\mathsf T}\widetilde H_{\mathrm C}.
```

Tissue initialization is:

```math
H_{\mathrm T}^{(0)}
=
\left[
H_{\mathrm T}
\middle\Vert
U_{\mathrm C\to T}
\right].
```

Tissue context is:

```math
\widetilde H_{\mathrm T}
=
\mathcal C_{\mathrm T}
\left(
H_{\mathrm T}^{(0)},A_{\mathrm T}
\right).
```

The graph-level readout is:

```math
z_{\mathrm{HACT}}
=
\mathcal R_{\mathrm T}
\left(
\widetilde H_{\mathrm T}
\right).
```

The assignment is part of G and R simultaneously: it defines parent geometry
and transfers a child statistic. The source HACT task is breast-cancer
subtyping over annotated tissue regions; generic WSI survival use is an
extension.

## 8. The hierarchy boundary as a statistic

The child-to-parent operator determines what survives:

| Boundary statistic | Formula | Preserves | Loses |
| --- | --- | --- | --- |
| sum | sum of child states | total burden and count scale | within-parent contrasts |
| mean | normalized sum | first moment | count and higher moments |
| attention | weighted first moment | learned task direction | alternative mixtures |
| max | coordinatewise extremes | witnesses | multiplicity and co-occurrence |
| MaxS or MAS | one selected child | sparse representative | all unselected child states |
| MaxMinS | positive and negative representatives | opposing extremes | distribution beyond two selections |
| AFS | Tier-1 weighted embedding | first moment within pseudo-bag | child identities and higher moments |
| [CLS] | learned nonlinear token | task-shaped window summary | child field not encoded by [CLS] |
| assignment sum | B-transpose H | parent-local total | parent-local nullspace |

A paper should report each boundary, not only the final slide head.

## 9. Composition non-commutativity

Let C-child be within-parent context and R-child be coarsening. In general:

```math
\mathcal R_{\mathrm{child}}
\left(
\mathcal C_{\mathrm{child}}(H,G)
\right)
\ne
\mathcal C_{\mathrm{parent}}
\left(
\mathcal R_{\mathrm{child}}(H),G_{\mathrm{parent}}
\right).
```

The right side may not be able to reconstruct child relations. For a parent
sum:

```math
B^{\mathsf T}AH
\ne
A_{\mathrm{parent}}B^{\mathsf T}H
```

unless A, B, and the parent operator satisfy a special commutation relation:

```math
B^{\mathsf T}A
=
A_{\mathrm{parent}}B^{\mathsf T}.
```

This is a strong condition. A hierarchy changes the operator order, not only
the resolution.

## 10. When hierarchy reduces to a flat model

A hierarchical model can be algebraically redundant under restrictive
conditions. For example:

1. every parent contains exactly one child;
2. the parent map is only a permutation;
3. no parent-level context acts;
4. the final readout is an invariant operator applied to the unchanged child
   field.

Then:

```math
H^{(1)}
=
PH^{(0)}
\Longrightarrow
\mathcal R(H^{(1)})
=
\mathcal R(H^{(0)}).
```

A hierarchy is not generally redundant when:

```text
parent identity changes context support
the coarsener is nonlinear or selective
coordinates enter before coarsening
parent count affects feature scale
coarse tokens are the only task input
cross-parent context acts after coarsening
```

The condition is algebraic, not based on the number of named levels.

## 11. Hierarchy versus distribution representation

A distribution representation tracks a measure over feature values:

```math
\widehat\mu
=
\frac{1}{N}
\sum_{a=1}^{N}\delta_{h_a}.
```

A hierarchy tracks a family of conditional measures:

```math
\widehat\mu_b
=
\frac{1}{|V_b|}
\sum_{a\in V_b}\delta_{h_a},
\qquad
b=1,\ldots,R.
```

The global measure is a mixture:

```math
\widehat\mu
=
\sum_b
\frac{|V_b|}{N}
\widehat\mu_b.
```

The hierarchy retains parent membership if the collection of parent-indexed
measures is kept. A global distribution loses that membership. Conversely, a
hierarchy that keeps only one mean per parent may lose distributional detail
that a global prototype or histogram retains.

## 12. Complexity matrix

Let N be the fine count, R the parent count, m-b the parent sizes, and d the
channel width.

| Operation | Cost |
| --- | --- |
| assignment transfer | order nnz(B) d |
| balanced local attention | order N-squared-d divided by R |
| unbalanced local attention | order d times sum of m-b-squared |
| parent attention | order R-squared-d |
| DTFD Tier 1 attention | order N d plus score-network cost |
| DTFD selection | order N for top-1 or top-k selection |
| DTFD Tier 2 attention | order R d plus score-network cost |
| HIPT local attention | order 256-squared-d per fixed parent window |
| HIPT slide attention | order R-i-squared-d |
| HACT graph context | order edge-count times d-squared |

Balanced hierarchy cost is:

```math
\mathcal O
\left(
\sum_{b=1}^{R}m_b^2d+R^2d
\right)
=
\mathcal O
\left(
\frac{N^2}{R}d+R^2d
\right)
```

when m-b equals N/R. The equality fails for skewed partitions.

## 13. Parent-size statistics

For a parent mean:

```math
\bar h_b
=
\frac{1}{m_b}
\sum_{a\in V_b}h_a.
```

If child states are independent with covariance Sigma:

```math
\mathrm{Cov}(\bar h_b)
=
\frac{\Sigma}{m_b}.
```

For a parent sum:

```math
s_b
=
\sum_{a\in V_b}h_a,
```

```math
\mathrm{Cov}(s_b)
=
m_b\Sigma.
```

A parent-level model can infer parent size from either feature scale or
variance. Parent occupancy should therefore be included in G/R diagnostics.

For an attention distribution alpha-ab, the effective child count is:

```math
m_{\mathrm{eff},b}
=
\frac{1}{
\sum_{a\in V_b}\alpha_{ab}^2
}.
```

A nominal parent of size 100 can have effective support one.

## 14. Supervision placement

A hierarchy can receive supervision at several levels:

```math
\mathcal L
=
\mathcal L_{\mathrm{slide}}
+
\sum_{\ell=0}^{L-1}
\lambda_\ell
\mathcal L_\ell^{\mathrm{aux}}.
```

DTFD uses inherited slide labels at pseudo-bag Tier 1:

```math
\mathcal L_{\mathrm{DTFD}}
=
\mathcal L_{\mathrm{Tier1}}
+
\mathcal L_{\mathrm{Tier2}}.
```

HIPT uses DINO teacher-student losses for lower-scale pretraining and a
downstream slide objective:

```math
\mathcal L_{\mathrm{HIPT}}
=
\mathcal L_{\mathrm{DINO,local}}
+
\mathcal L_{\mathrm{DINO,region}}
+
\mathcal L_{\mathrm{downstream}}.
```

HACT uses graph-level task supervision, with pseudo-types and assignments
generated upstream rather than necessarily supervised by the slide label.

The same hierarchy can learn different representations under different S.

## 15. Boundary gradients

For a linear sum boundary:

```math
H_{\mathrm{parent}}
=
B^{\mathsf T}H,
```

the Jacobian is:

```math
\frac{\partial\mathrm{vec}(H_{\mathrm{parent}})}
{\partial\mathrm{vec}(H)}
=
B\otimes I_d.
```

Its rank is:

```math
\mathrm{rank}(B)\,d.
```

The nullspace dimension is:

```math
\left(
N-\mathrm{rank}(B)
\right)d.
```

For a two-level linearized hierarchy:

```math
z
=
W_2
A_2
B^{\mathsf T}
A_1H,
```

the child gradient is:

```math
\frac{\partial z}{\partial\mathrm{vec}(H)}
=
W_2
\left(
A_2B^{\mathsf T}A_1
\right)
\otimes I_d
```

schematically, with channel transforms absorbed into W-2. A child direction
can be invisible because it lies in the nullspace of B-transpose, A-1, or the
final head.

## 16. Attribution across levels

For an output q and child a, full attribution is:

```math
\frac{\partial q}{\partial h_a}
=
\sum_b
\frac{\partial q}{\partial h_b^{(\mathrm{parent})}}
\frac{\partial h_b^{(\mathrm{parent})}}{\partial h_a}.
```

If parent context mixes parents:

```math
h_b^{+}
=
\mathcal C_{\mathrm{parent}}
\left(
\{h_c\}_{c=1}^{R}
\right)_b,
```

then every parent reachable through context can contribute to child a's
derivative. The product of local attention weights is not generally the full
gradient.

A perturbation attribution is:

```math
\Delta_a^{(q)}
=
q(\mathcal H)
-
q(\mathcal H\setminus a),
```

but deletion must state whether parent membership, parent tokens, and parent
context are rebuilt.

## 17. Equivariance conditions

Let Q permute children and S permute parents. A hard parent construction is
equivariant when:

```math
B(QH)
=
QB(H)S^{\mathsf T}.
```

A child context is equivariant when:

```math
\mathcal C_{\mathrm{child}}(QH,QGQ^{\mathsf T})
=
Q\mathcal C_{\mathrm{child}}(H,G).
```

The parent representation is then:

```math
B(QH)^{\mathsf T}
\mathcal C_{\mathrm{child}}(QH,QGQ^{\mathsf T})
=
S
B(H)^{\mathsf T}
\mathcal C_{\mathrm{child}}(H,G).
```

A parent readout invariant to S gives a slide-level invariant. Failure of this
identity means the hierarchy includes an order-dependent construction.

## 18. Design axes exposed by the matrix

### 18.1 Distribution-preserving boundaries

Add parent moments:

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

This preserves mean, dispersion, and count but increases estimation variance and
dimension.

### 18.2 Soft assignment uncertainty

Use routing probabilities pi-ab:

```math
\pi_{ab}\ge0,
\qquad
\sum_b\pi_{ab}=1.
```

Transfer becomes:

```math
H_{\mathrm{parent}}
=
\Pi^{\mathsf T}H.
```

Uncertainty is retained, but parent semantics become expected rather than hard.

### 18.3 Cross-parent child memory

Retain a limited child residual:

```math
z
=
\mathcal H
\left(
H_{\mathrm{parent}},
\mathcal R_{\mathrm{child}}(H)
\right).
```

This reduces boundary nullspace but changes the architecture from strict
coarsening to a skip-connected hierarchy.

### 18.4 Task-conditioned hierarchy

Let parent assignment depend on a task query q:

```math
\pi_{ab}(q)
=
\mathrm{softmax}_b
\left(
r_\theta(h_a,q)
\right).
```

This can retain task-relevant regions but creates a stronger supervision and
interpretability dependence.

### 18.5 Survival-specific scale readouts

Use horizon-specific parent summaries:

```math
z_i^{(k)}
=
\mathcal R_{\theta,k}
\left(
H_i^{(L)}
\right),
\qquad
\eta_i^{(k)}
=
w_k^{\mathsf T}z_i^{(k)}.
```

This allows different scales to matter at different horizons but increases
multiple-readout complexity.

## 19. Paper comparison matrix

| Question | DTFD-MIL | HIPT | HACT |
| --- | --- | --- | --- |
| What is a parent? | random pseudo-bag | fixed spatial window | tissue region |
| Is membership anatomical? | no | computationally spatial | explicitly cell-to-tissue |
| What is C? | gated attention at two tiers | transformer context at three scales | graph context at cell and tissue scales |
| What is R? | selection/AFS then attention | [CLS] extraction | assignment sum then tissue readout |
| What survives? | selected or weighted pseudo-bag features | learned window tokens | coarse tissue summaries |
| Main nullspace | unselected patches | [CLS] collisions | assignment contrasts |
| Main S | inherited slide labels | DINO and downstream task | task label with generated graph attributes |
| Main risk | false positive pseudo-bag labels | window and scale mismatch | segmentation and assignment error |

## 20. Reporting template

A hierarchical MIL paper should report:

```text
1. fine unit, scale, and feature dimension
2. number of levels and parent count at every level
3. exact parent construction and whether it is label-free
4. within-level and cross-level context operators
5. boundary readout formula and retained metadata
6. occupancy, empty-parent, and padding conventions
7. parent-size distribution and effective sample size
8. auxiliary labels and their provenance
9. exact task head and loss
10. patient-level split before any learned partition or index construction
11. boundary ablations, assignment perturbations, and readout swaps
12. compute and memory costs under actual unbalanced partitions
```

The C/R/G/S record is:

```math
\begin{aligned}
G &: \{P_\ell,G^{(\ell)}\}_{\ell=0}^{L-1},
\\
C &: \{\mathcal C_\ell\}_{\ell=0}^{L},
\\
R &: \{\mathcal R_\ell\}_{\ell=0}^{L},
\\
S &: \{\mathcal L_{\ell}^{\mathrm{aux}},\mathcal L_{\mathrm{slide}}\}.
\end{aligned}
```

## 21. Final synthesis

A hierarchy is:

```math
\boxed{
\text{fine object}
\longrightarrow
\text{boundary statistic}
\longrightarrow
\text{coarse context}
\longrightarrow
\text{boundary statistic}
\longrightarrow
\text{slide head}
}
```

DTFD, HIPT, and HACT occupy different points in the design space:

```math
\begin{aligned}
\mathrm{DTFD}
&:
\text{random partition}
+
\text{selection}
+
\text{two-tier attention},
\\
\mathrm{HIPT}
&:
\text{nested spatial geometry}
+
\text{transformer context}
+
\text{CLS summaries},
\\
\mathrm{HACT}
&:
\text{explicit parent graph}
+
\text{assignment transfer}
+
\text{coarse graph context}.
\end{aligned}
```

The useful question is not whether one has “more hierarchy.” It is:

```math
\boxed{
\text{which task-relevant statistic is retained at each boundary, and what
geometry and supervision make that statistic valid?}
}
```
