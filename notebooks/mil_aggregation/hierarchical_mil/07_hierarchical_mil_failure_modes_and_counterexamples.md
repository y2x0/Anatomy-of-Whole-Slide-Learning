# Hierarchical MIL Failure Modes and Counterexamples

Hierarchical MIL replaces one flat bag with a sequence of boundary operators:

```math
\mathcal B_i^{(0)}
\xrightarrow{\mathcal C_0,\mathcal R_0,P_0}
\mathcal B_i^{(1)}
\xrightarrow{\mathcal C_1,\mathcal R_1,P_1}
\cdots
\xrightarrow{\mathcal R_L}
z_i.
```

The parent map P can be random, spatial, graph-derived, or learned. The
context operator C can be attention, a transformer, a graph network, or a
state-space scan. The coarsener R can be a sum, mean, attention, selection,
[CLS] token, or a learned router.

A hierarchy fails when a boundary removes a task-relevant distinction, assigns
evidence to the wrong parent, or creates a shortcut through occupancy or
sampling. The right unit of analysis is each boundary, not the word
“hierarchical.”

## 1. Source anchors

The paper-specific examples use:

| Method | Source note | Boundary analyzed |
| --- | --- | --- |
| DTFD-MIL | [pseudo-bags and two-tier distillation](03_dtfd_pseudo_bags_and_two_tier_distillation.md) | random pseudo-bag partition, Tier-1 selection, Tier-2 attention |
| HIPT | [nested tokens and multiscale readout](04_hipt_nested_tokens_and_multiscale_readout.md) | nested spatial windows and [CLS] extraction |
| HACT | [hierarchical graph boundary](05_hact_and_hierarchical_graph_mil_boundary.md) | cell graph, assignment matrix, tissue graph |

These methods share a fine-to-coarse pattern but do not share the same parent
semantics.

## 2. Boundary map and information loss

At level ell, let the child field be:

```math
H^{(\ell)}
\in
\mathbb R^{N_\ell\times d_\ell}.
```

Let P-ell map children to M-ell parents. A linear transfer is:

```math
T_\ell
=
P_\ell^{\mathsf T}H^{(\ell)}.
```

If P-ell is a hard assignment matrix, every child belongs to one parent:

```math
P_\ell\in\{0,1\}^{N_\ell\times M_\ell},
\qquad
P_\ell\mathbf 1_{M_\ell}
=
\mathbf 1_{N_\ell}.
```

A nonlinear or attention boundary is:

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

The boundary has a collision whenever:

```math
H^{(\ell)}\ne H^{(\ell)\prime},
\qquad
H^{(\ell+1)}
=
H^{(\ell+1)\prime}.
```

No later level can recover a distinction that is absent from its input unless a
skip path, auxiliary loss, or retained metadata carries it forward.

## 3. Coarsening nullspace

For linear assignment transfer:

```math
T_\ell(H)
=
P_\ell^{\mathsf T}H.
```

Any perturbation Delta in the nullspace satisfies:

```math
P_\ell^{\mathsf T}\Delta=0.
```

Then:

```math
T_\ell(H+\Delta)
=
T_\ell(H).
```

If every parent contains at least two children, within-parent contrasts often
generate nonzero nullspace directions. For one scalar parent containing two
children:

```math
P
=
\begin{bmatrix}
1\\
1
\end{bmatrix},
\qquad
\Delta
=
\begin{bmatrix}
\delta\\
-\delta
\end{bmatrix},
```

```math
P^{\mathsf T}\Delta
=
0.
```

The parent receives the same sum from:

```math
H
=
\begin{bmatrix}
u+\delta\\
u-\delta
\end{bmatrix}
\qquad
\text{and}
\qquad
H'
=
\begin{bmatrix}
u\\
u
\end{bmatrix}.
```

A parent-level model cannot distinguish concentration from uniformity unless
the boundary also retains variance, a max, a histogram, a child token, or a
skip connection.

## 4. Partition mismatch

Suppose four fine instances admit two partitions:

```math
P_A:
\quad
\{x_1,x_2\},
\{x_3,x_4\},
```

```math
P_B:
\quad
\{x_1,x_3\},
\{x_2,x_4\}.
```

If the label depends on the interaction between x-1 and x-3, P-B places the
relevant pair under one parent while P-A separates it. A within-parent
operator followed by parent pooling can therefore distinguish the two
partition choices even though the flat child multiset is identical.

The error is representational:

```math
I(x_1;x_3\mid\text{same parent})
\ne
I(x_1;x_3\mid\text{different parents}).
```

A later coarse context can connect the parents, but only if their summaries
retain the information needed to identify the pair.

## 5. DTFD pseudo-bag label noise

Let a positive slide contain n patches, k positive patches, and let a random
pseudo-bag contain m patches sampled without replacement. The probability that
the pseudo-bag contains no positive patch is exactly:

```math
\Pr(K_{\mathrm{pos}}=0\mid Y=1)
=
\frac{\binom{n-k}{m}}{\binom{n}{m}},
\qquad
m\le n-k.
```

If p=k/n and sampling is approximated as independent:

```math
\Pr(K_{\mathrm{pos}}=0\mid Y=1)
\approx
(1-p)^m.
```

DTFD inherits the slide label for every pseudo-bag:

```math
Y_r=Y.
```

Therefore a positive pseudo-bag can be locally negative. The Tier-1
cross-entropy contains label noise whose rate increases as m decreases:

```math
\mathcal L_1
=
-\sum_{r=1}^{R}
\left[
Y\log q_r
+
(1-Y)\log(1-q_r)
\right].
```

Increasing the number of pseudo-bags R is not free sample multiplication.
Since m is approximately n/R, R changes both optimization sample count and
pseudo-bag label reliability.

## 6. DTFD selection collisions

Let Tier-1 attention produce:

```math
a_{rj}
=
\frac{\exp(s_{rj})}
{\sum_{u\in B_r}\exp(s_{ru})},
\qquad
z_r
=
\sum_{j\in B_r}a_{rj}h_{rj}.
```

For class c, the operational instance signal is:

```math
g_{rj,c}
=
a_{rj}
w_c^{\mathsf T}h_{rj}.
```

MaxS selects:

```math
j_r^{\mathrm{MaxS}}
=
\underset{j\in B_r}{\arg\max}
p_{rj,1}.
```

MAS selects:

```math
j_r^{\mathrm{MAS}}
=
\underset{j\in B_r}{\arg\max}
a_{rj}.
```

These can differ because attention is not class direction:

```math
\underset{j}{\arg\max}\;a_{rj}
\ne
\underset{j}{\arg\max}\;
a_{rj}w_1^{\mathsf T}h_{rj}.
```

MaxMinS retains two selected states, while AFS retains z-r. Each rule creates
a different collision class. A later Tier-2 model cannot recover unselected
patch identities under MaxS, MAS, or MaxMinS.

## 7. Positive-instance selection recall

Let P-r be the set of truly positive instances in pseudo-bag r and S-r the
selected set. Define selection recall:

```math
\mathrm{Recall}_{r}
=
\frac{|P_r\cap S_r|}{|P_r|}
```

when P-r is nonempty. For a top-k selector:

```math
\mathrm{Recall}_{r,k}
=
\frac{
|P_r\cap\mathrm{TopK}_r|
}{
|P_r|
}.
```

A high slide-level score can coexist with low selection recall if the model
uses a slide shortcut or if one selected patch is sufficient for classification.
Selection recall is therefore a representation diagnostic, not merely an
interpretability metric.

## 8. HIPT [CLS] boundary

For a parent window with child tokens X:

```math
X
\in
\mathbb R^{m\times d},
\qquad
c
=
\mathrm{ViT}_{\theta}(X)_{\mathrm{CLS}}.
```

The [CLS] vector c is a learned nonlinear summary. It is not the mean:

```math
c
\ne
\frac{1}{m}
\sum_{j=1}^{m}x_j
```

in general.

Two child fields can produce the same [CLS] token:

```math
X\ne X',
\qquad
\mathrm{ViT}_{\theta}(X)_{\mathrm{CLS}}
=
\mathrm{ViT}_{\theta}(X')_{\mathrm{CLS}}.
```

The next scale receives the same parent token and cannot distinguish the
fields through that path. Positional embeddings make the summary sensitive to
within-window arrangement, but they do not preserve the complete arrangement.

The hierarchy therefore changes the object from a child set or sequence to a
set or sequence of learned parent tokens.

## 9. HIPT nested geometry

At cell, region, and slide scales, write:

```math
H^{(0)}
=
\{x_{rpc}\},
\qquad
H^{(1)}
=
\{h_{rp}\},
\qquad
H^{(2)}
=
\{h_r\},
\qquad
z
=
h_{\mathrm{slide}}.
```

The parent maps are spatial and non-overlapping by construction. A boundary
error can arise from:

```text
window alignment
tissue masking
padding at region boundaries
missing child tokens
positional-coordinate mismatch
```

If a morphology motif crosses a fixed window boundary, no lower-level [CLS]
token contains the complete motif. The region-level transformer can only
combine the partial summaries it receives.

## 10. HACT assignment errors

Let the hard cell-to-tissue assignment matrix be:

```math
B\in\{0,1\}^{N_{\mathrm C}\times N_{\mathrm T}},
\qquad
\sum_bB_{ab}=1.
```

The transfer is:

```math
U_{\mathrm C\to T}
=
B^{\mathsf T}\widetilde H_{\mathrm C}.
```

If cell a is moved from tissue b to tissue b-prime, the assignment perturbation
is:

```math
\Delta B
=
e_a
\left(
e_{b'}-e_b
\right)^{\mathsf T}.
```

The transferred state changes by:

```math
\Delta U
=
\Delta B^{\mathsf T}\widetilde H_{\mathrm C}
=
\left(
e_{b'}-e_b
\right)
\widetilde h_{\mathrm C,a}^{\mathsf T}.
```

The total sum across tissue nodes is unchanged:

```math
\mathbf 1^{\mathsf T}\Delta U=0.
```

But the signal is relocated to another tissue node, and the tissue graph can
propagate that error.

## 11. Hierarchy-induced order dependence

A parent map derived from a permutation-sensitive algorithm can change when the
same children are reordered. For a set-consistent construction:

```math
\mathcal P(QH)
=
Q_{\mathrm{parent}}\mathcal P(H)
```

for every child permutation Q, with an induced parent permutation. If this
identity fails, the model has an unreported order geometry in addition to its
declared hierarchy.

A learned router can be permutation-equivariant while still being unstable
under small feature perturbations. Both properties should be tested separately.

## 12. Learned routing collapse

Let router probabilities be:

```math
\pi_{ab}
=
\frac{\exp(r_{ab})}
{\sum_{c=1}^{R}\exp(r_{ac})}.
```

The entropy of child a's routing is:

```math
H_a
=
-\sum_{b=1}^{R}\pi_{ab}\log\pi_{ab}.
```

Two collapse modes are:

```math
\pi_{ab}\approx1
\quad
\text{for one parent b},
```

and:

```math
\pi_{ab}\approx\frac{1}{R}
\quad
\text{for all b}.
```

The first creates parent under-utilization. The second erases semantic parent
identity and approximates a global average.

Parent occupancy is:

```math
o_b
=
\frac{1}{N}
\sum_{a=1}^{N}\pi_{ab}.
```

An effective parent count is:

```math
R_{\mathrm{eff}}
=
\exp
\left(
-\sum_{b=1}^{R}o_b\log o_b
\right).
```

Report occupancy and routing entropy rather than assuming that the declared R
parents are all used.

## 13. Boundary attention and attribution

Suppose a child-to-parent boundary uses local coefficients alpha-a-p and a
parent-level readout uses beta-p. The naive effective child weight is:

```math
\omega_a
=
\beta_{p(a)}
\alpha_{a,p(a)}.
```

This is valid only for a linear composition with fixed attention weights and no
context between levels. If the parent transformer changes parent states:

```math
h_p'
=
\mathcal C_{\mathrm{parent}}
\left(
\{h_q\}_q
\right)_p,
```

then child a can influence many parents through the Jacobian:

```math
\frac{\partial z}{\partial h_a}
=
\sum_p
\frac{\partial z}{\partial h_p'}
\frac{\partial h_p'}{\partial h_a}.
```

A parent heatmap is not a child explanation. A child attention map is not a
full output attribution. A deletion explanation must recompute every affected
boundary.

## 14. Coarse context can amplify local errors

Let the coarse context be linearized:

```math
H^{(2)}
=
\widetilde A_2
\left(
P^{\mathsf T}
\widetilde A_1H^{(0)}
\right).
```

An assignment perturbation Delta-P produces:

```math
\Delta H^{(2)}
=
\widetilde A_2
\Delta P^{\mathsf T}
\widetilde A_1H^{(0)}.
```

A norm bound is:

```math
\left\|
\Delta H^{(2)}
\right\|_{\mathrm F}
\le
\left\|
\widetilde A_2
\right\|_2
\left\|
\Delta P
\right\|_2
\left\|
\widetilde A_1H^{(0)}
\right\|_{\mathrm F}.
```

The coarse graph can spread one parent assignment error across neighboring
parents. More hierarchy is therefore not automatically more robust.

## 15. Flat-set versus hierarchy collision

Let two child fields have equal global mean:

```math
\frac{1}{N}
\sum_{a=1}^{N}h_a
=
\frac{1}{N'}
\sum_{a=1}^{N'}h_a'.
```

A flat mean model collides. A hierarchy can distinguish them if their parent
means differ:

```math
\left\{
\frac{1}{|V_b|}
\sum_{a\in V_b}h_a
\right\}_{b}
\ne
\left\{
\frac{1}{|V_b'|}
\sum_{a\in V_b'}h_a'
\right\}_{b}.
```

The reverse can also happen. Two fields can have identical parent means while
their child distributions differ within every parent. The hierarchy has
preserved local organization but discarded within-parent dispersion.

## 16. Same hierarchy, different coarsener

Fix parent groups V-b and compare sum and mean:

```math
z_b^{\mathrm{sum}}
=
\sum_{a\in V_b}h_a,
\qquad
z_b^{\mathrm{mean}}
=
\frac{1}{|V_b|}
\sum_{a\in V_b}h_a.
```

If one parent has twice as many children with the same mean, sum doubles while
mean does not. The hierarchy's geometry is identical; R changes the surviving
burden statistic.

A second-level attention readout can create a third count effect because parent
scores are normalized over the number of parents, not the number of children.

## 17. Complexity is conditional on partition balance

For N children, R parents, and balanced parent size m=N/R, local dense attention
plus parent dense attention has cost:

```math
\mathcal O(Rm^2d+R^2d).
```

Substituting m=N/R gives:

```math
\mathcal O
\left(
\frac{N^2}{R}d+R^2d
\right).
```

This is less than flat N-squared-d attention only for favorable R and balanced
groups. If one group has size m-max:

```math
\mathcal O
\left(
\sum_bm_b^2d+R^2d
\right)
\ge
\mathcal O(m_{\max}^2d+R^2d).
```

A single oversized parent can dominate compute and memory. Padding all parents
to m-max makes the implementation cost:

```math
\mathcal O(Rm_{\max}^2d+R^2d).
```

The theoretical balanced cost is not the runtime guarantee.

## 18. Boundary sampling and variance

If a parent mean uses m children with independent covariance Sigma:

```math
\mathrm{Cov}
\left(
\frac{1}{m}\sum_{a=1}^{m}h_a
\right)
=
\frac{\Sigma}{m}.
```

Small parents have higher sampling variance. For a sum:

```math
\mathrm{Cov}
\left(
\sum_{a=1}^{m}h_a
\right)
=
m\Sigma.
```

A hierarchy with highly unequal parent sizes therefore creates heteroscedastic
parent tokens. A later attention layer can learn parent size through both
feature scale and score behavior.

For a weighted parent mean with weights alpha-a:

```math
z_b
=
\sum_{a\in V_b}\alpha_{ab}h_a,
\qquad
\sum_{a\in V_b}\alpha_{ab}=1,
```

the effective sample size is:

```math
m_{\mathrm{eff},b}
=
\frac{1}{
\sum_{a\in V_b}\alpha_{ab}^2
}.
```

A parent with many children can still have effective sample size near one.

## 19. Geometry leakage and split leakage

A hierarchy is not weakly supervised if the parent map uses outcome information.
Potential leakage includes:

```text
tumor masks derived from labels
patient outcomes used to choose regions
global learned routing fit before patient splitting
overlapping windows shared across train and test
cross-slide indices built before the split
```

The split condition should hold for all derived hierarchy objects:

```math
\mathcal D_{\mathrm{train}}\cap\mathcal D_{\mathrm{test}}=\varnothing
```

at the patient level, and:

```math
P_{\mathrm{train}}
=
\Gamma_{\mathrm P}
\left(
\mathcal D_{\mathrm{train}}
\right),
\qquad
P_{\mathrm{test}}
=
\Gamma_{\mathrm P}
\left(
\mathcal D_{\mathrm{test}}
\right).
```

Coordinates alone are not leakage. A label-dependent partition is supervision
and must be declared as such.

## 20. Boundary deletion interventions

Define full deletion of child a:

```math
\Delta_a^{\mathrm{full}}
=
q(\mathcal B)
-
q(\mathcal B\setminus a).
```

Define fixed-parent deletion, where the parent map remains unchanged:

```math
\Delta_a^{\mathrm{fixedP}}
=
q(P,\mathcal B)
-
q(P,\mathcal B\setminus a).
```

Define rebuilt-parent deletion:

```math
\Delta_a^{\mathrm{rebuildP}}
=
q(P(\mathcal B),\mathcal B)
-
q(P(\mathcal B\setminus a),\mathcal B\setminus a).
```

The difference between fixed and rebuilt interventions measures the effect of
changing the hierarchy itself:

```math
\Delta_a^{\mathrm{geometry}}
=
\Delta_a^{\mathrm{fixedP}}
-
\Delta_a^{\mathrm{rebuildP}}.
```

For DTFD, deleting a patch can change pseudo-bag membership and Tier-1
selection. For HIPT, deleting a token can alter [CLS] states at one or more
scales. For HACT, deleting a cell changes assignment sums and graph messages.

## 21. Counterexample: boundary destroys a pair

Let the label depend on the existence of a pair with complementary features:

```math
Y
=
\mathbf 1
\left\{
\exists(a,b):
h_a=1,\ h_b=-1,\ a\sim b
\right\}.
```

Suppose one hierarchy places the pair under the same parent and another places
them in separate parents. If each parent retains only its mean, the separated
hierarchy can produce two parent means that do not reveal the pair relation.
A parent-level mean cannot infer an unobserved cross-parent pairing without
additional child identities or cross-parent context.

## 22. Counterexample: equal parent sums

Take one scalar parent with:

```math
H
=
\begin{bmatrix}
u+\delta\\
u-\delta
\end{bmatrix},
\qquad
H'
=
\begin{bmatrix}
u\\
u
\end{bmatrix}.
```

Then:

```math
\mathbf 1^{\mathsf T}H
=
\mathbf 1^{\mathsf T}H'
=
2u.
```

Any later model receiving only the parent sum collides. A parent variance would
separate them:

```math
\mathrm{Var}(H)
=
\delta^2,
\qquad
\mathrm{Var}(H')=0.
```

## 23. Counterexample: [CLS] arrangement collision

Let X and X-prime be two token arrangements with the same multiset but
different order. A transformer with positional embeddings can produce:

```math
\mathrm{CLS}(X)
\ne
\mathrm{CLS}(X').
```

But the [CLS] map is many-to-one in high-dimensional token space, so there can
also be:

```math
X\ne X',
\qquad
\mathrm{CLS}(X)
=
\mathrm{CLS}(X').
```

The next hierarchy level cannot recover the difference unless a token-level skip
or auxiliary representation is retained.

## 24. Counterexample: inherited positive pseudo-label

Let p be the positive-patch fraction and m pseudo-bag size. The probability of
a false positive inherited label is:

```math
\Pr(\text{no positive patch}\mid Y=1)
\approx
(1-p)^m.
```

For p equal to 0.01 and m equal to 50, this probability is approximately
0.605. Thus more pseudo-bags can create many locally negative examples labeled
positive. Tier-1 distillation may still work as a regularizer, but the
pseudo-bag label should not be interpreted as an observed local label.

## 25. Counterexample: assignment error with coarse propagation

Let one cell state h be assigned to tissue one in the correct hierarchy and
tissue two in the perturbed hierarchy:

```math
U
=
\begin{bmatrix}
h\\
0
\end{bmatrix},
\qquad
U'
=
\begin{bmatrix}
0\\
h
\end{bmatrix}.
```

If tissue propagation is:

```math
H_{\mathrm T}^{(1)}
=
\widetilde A_{\mathrm T}U,
```

then:

```math
H_{\mathrm T}^{(1)}-
H_{\mathrm T}^{(1)\prime}
=
\widetilde A_{\mathrm T}
\begin{bmatrix}
h\\
-h
\end{bmatrix}.
```

A tissue adjacency can spread the parent assignment error to every reachable
node.

## 26. Failure-mode matrix

| Method | Geometry failure | Context failure | Readout failure | Supervision/deployment failure |
| --- | --- | --- | --- | --- |
| DTFD | random pseudo-bags do not equal tissue regions | inherited Tier-1 labels can be locally false | selection loses unselected patches; AFS retains only first moment | pseudo-bag count, class imbalance, selection shortcut |
| HIPT | fixed windows cut motifs and inherit masking artifacts | local [CLS] bottlenecks and positional assumptions | [CLS] collisions at every scale | pretraining/downstream domain and scale shift |
| HACT | wrong cell-region assignment or graph support | fine errors propagate through tissue graph | assignment sums lose within-region contrasts | segmentation quality and region-scale shift |
| generic hierarchy | parent map is unstable or label-dependent | context acts on already compressed parents | boundary statistic mismatch | occupancy, leakage, and unequal parent variance |

## 27. C/R/G/S failure checklist

Before attributing a gain to hierarchy, record:

```math
\begin{aligned}
G &: \text{what defines parent membership and is it fixed or learned?}
\\
C &: \text{what context occurs before and after each boundary?}
\\
R &: \text{what statistic survives each boundary?}
\\
S &: \text{which labels supervise children, parents, and the slide?}
\end{aligned}
```

Run at least:

```text
balanced versus adversarial partitions
parent occupancy and effective sample size
boundary perturbation and assignment shuffle
selection recall for sparse positives
[CLS] or parent-token deletion
fixed-parent versus rebuilt-parent deletion
patient-level leakage audit
readout swaps at every level
```

## 28. Bottom line

A hierarchy is not automatically more informative than a flat bag:

```math
\boxed{
\text{fine evidence}
\longrightarrow
\text{boundary statistic}
\longrightarrow
\text{coarse context}
\longrightarrow
\text{slide statistic}
}
```

Every boundary creates a possible nullspace. Every parent map creates a geometry
assumption. Every inherited or auxiliary label changes S.

The correct question is:

```math
\boxed{
\text{which task-relevant distinctions survive each child-to-parent boundary,
and which errors does the hierarchy amplify?}
}
```
