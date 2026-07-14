# Patch, Slide, And Multimodal Object Alignment

Primary anchors:

- [HIPT](https://arxiv.org/abs/2206.02647)
- [UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/)
- [Virchow](https://arxiv.org/abs/2309.07778)
- [PLIP](https://pubmed.ncbi.nlm.nih.gov/37592105/)
- [CONCH](https://arxiv.org/abs/2307.12914)
- [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w)
- [TITAN](https://www.nature.com/articles/s41591-025-03982-3)
- [KEEP](https://arxiv.org/html/2412.13126v1)
- [patch-versus-slide derivation](../paper_specific_derivations/09_patch_vs_slide_pretraining_statistics.md)
- [transfer and readout limits](../paper_specific_derivations/10_foundation_model_transfer_and_readout_limits.md)

Patch, slide, hierarchy, text, and retrieval models do not merely use different
architectures. They represent different random objects. Comparing them requires
asking whether one representation factors through another and whether the
corresponding diagram commutes.

## 1. Four Source Objects

Let a whole slide be a coordinate-indexed collection:

```math
\mathcal X_i
=
\left\{
    (x_{ij},c_{ij})
    :
    j=1,\ldots,n_i
\right\}.
```

Four derived objects are common.

The patch multiset is:

```math
\mathsf B_i
=
\{x_{ij}\}_{j=1}^{n_i}.
```

The empirical patch measure is:

```math
\widehat\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\delta_{x_{ij}}.
```

The coordinate sequence under serialization `s` is:

```math
\mathsf Q_i^{(s)}
=
\left[
    (x_{i,s(1)},c_{i,s(1)}),
    \ldots,
    (x_{i,s(n_i)},c_{i,s(n_i)})
\right].
```

The spatial graph is:

```math
\mathsf G_i
=
\left(
    V_i,
    E_i,
    X_i,
    C_i
\right).
```

The maps from the full coordinate-indexed slide to these objects are
many-to-one. For example:

```math
\mathcal X_i
\longmapsto
\widehat\mu_i
```

removes layout while retaining the empirical distribution.

## 2. Patch Pretraining Followed By Aggregation

Let the patch encoder be:

```math
h_{ij}
=
f_{\phi}(x_{ij})
\in
\mathbb R^d.
```

The patch-pretrained slide pipeline is:

```math
z_i^{\mathrm{patch}}
=
\mathcal R_{\psi}
\left(
    \mathcal C_{\psi}
    \left(
        \{h_{ij},c_{ij}\}_{j=1}^{n_i}
    \right)
\right).
```

Write this composite as:

```math
A_{\psi}\circ f_{\phi}^{\otimes n_i}.
```

The tensor-power notation means the same patch map is applied to every patch;
it does not imply an ordered tensor when the downstream object is a set.

UNI and Virchow usually enter WSI systems through this boundary. Their
pretraining organizes `f_phi`; the chosen downstream operator `A_psi` decides
which slide statistic survives.

## 3. Direct Slide Pretraining

A slide encoder can act on patches and geometry jointly:

```math
z_i^{\mathrm{slide}}
=
F_{\theta}
\left(
    \mathcal X_i
\right).
```

For coordinate-aware context:

```math
\widetilde h_{ij}
=
\mathcal C_{\theta}
\left(
    h_{ij},
    \{h_{ir},c_{ir}\}_{r=1}^{n_i}
\right),
```

```math
z_i^{\mathrm{slide}}
=
\mathcal R_{\theta}
\left(
    \{\widetilde h_{ij}\}_{j=1}^{n_i}
\right).
```

Prov-GigaPath and TITAN train slide-level context operators before downstream
task fitting. HIPT trains local and regional representations before a
task-adapted WSI module.

## 4. Exact Commuting-Diagram Criterion

Let patch-derived representation be:

```math
P(X)
=
A_{\psi}
\left(
    \{f_{\phi}(x_j),c_j\}_{j=1}^{n}
\right).
```

Let direct slide representation be:

```math
S(X)
=
F_{\theta}(X).
```

The diagram commutes when there exists a transfer map `T` such that:

```math
\boxed{
S
=
T\circ P.
}
```

At the set-theoretic level, this factorization exists if and only if `S` is
constant on every fiber of `P`:

```math
\boxed{
P(X)
=
P(X')
\Longrightarrow
S(X)
=
S(X').
}
```

The forward direction is immediate. For the reverse direction, define `T(p)`
as `S(X)` for any `X` satisfying `P(X)=p`. Fiber constancy makes this definition
well-defined on the image of `P`.

Measurability is a separate requirement. In the finite empirical setting it is
automatic. For general measurable spaces, the fiber argument guarantees only a
set-theoretic factorization; a measurable `T` additionally requires the
sigma-algebra on the image of `P` to support the induced quotient map. The
usual standard-Borel formulation therefore states the criterion with that
regularity assumption, or restricts `T` to the image of `P` equipped with the
final sigma-algebra:

```math
\Sigma_{P}
=
\left\{
    B\subseteq P(\mathcal X)
    :
    P^{-1}(B)\in\Sigma_{\mathcal X}
\right\}.
```

Under this sigma-algebra, fiber constancy and measurability of `S` make the
induced transfer measurable. For every measurable target set `C`:

```math
P^{-1}
\left(
    T^{-1}(C)
\right)
=
S^{-1}(C)
\in
\Sigma_{\mathcal X},
```

so `T^{-1}(C)` belongs to `\Sigma_P` by definition. If the image of `P` is
instead equipped with the Borel sigma-algebra inherited from an ambient
Euclidean space, that conclusion can require additional quotient regularity.

Thus patch aggregation can reproduce a direct slide representation exactly
only when every collision created by patch encoding and aggregation is also a
collision of the slide encoder.

## 5. Approximate Commutation Error

Assume that `S(X)` is square integrable. When exact factorization is
unrealistic, define:

```math
\mathcal E_{\mathrm{comm}}
=
\inf_T
\mathbb E
\left[
    \left\|
        S(X)
        -
        T(P(X))
    \right\|_2^2
\right].
```

Under squared loss, the optimal transfer map is:

```math
T^{\star}(p)
=
\mathbb E
\left[
    S(X)
    \mid
    P(X)=p
\right].
```

The irreducible commutation error is:

```math
\boxed{
\mathcal E_{\mathrm{comm}}
=
\mathbb E
\left[
    \mathrm{tr}
    \mathrm{Cov}
    \left(
        S(X)
        \mid
        P(X)
    \right)
\right].
}
```

This is the slide-representation variation remaining inside patch-pipeline
fibers. A stronger transfer head cannot reduce it below the conditional
variance unless `P` itself changes.

## 6. Task-Level Commutation Is Weaker

Exact representation commutation is stronger than needed for one task. Assume
that the scalar target `Y` is square integrable, and let the squared-loss Bayes
rule be:

```math
m_Y(X)
=
\mathbb E[Y\mid X].
```

Patch representation `P` is sufficient for the task when:

```math
m_Y(X)
=
g(P(X))
```

for some `g`.

Equivalently under squared loss:

```math
\mathbb E
\left[
    \mathrm{Var}
    \left(
        m_Y(X)
        \mid
        P(X)
    \right)
\right]
=
0.
```

Therefore a patch pipeline can match slide-task performance without reproducing
the direct slide embedding. It needs to preserve the task functional, not every
coordinate of `S(X)`.

This distinction prevents two opposite errors:

```text
embedding mismatch does not prove task insufficiency;
task parity on one label does not prove representation equivalence.
```

## 7. Matched Marginal, Different Joint Counterexample

Let each slide contain two binary patch states at left and right coordinates:

```math
X
=
(X_L,X_R).
```

Population `A` contains aligned states:

```math
P_A(0,0)
=
P_A(1,1)
=
\frac{1}{2}.
```

Population `B` contains anti-aligned states:

```math
P_B(0,1)
=
P_B(1,0)
=
\frac{1}{2}.
```

Both have identical patch marginals:

```math
P_A(X_L=1)
=
P_B(X_L=1)
=
\frac{1}{2},
```

```math
P_A(X_R=1)
=
P_B(X_R=1)
=
\frac{1}{2}.
```

Their joint structures differ:

```math
\mathbb E_A[X_LX_R]
=
\frac{1}{2},
\qquad
\mathbb E_B[X_LX_R]
=
0.
```

A representation retaining only the empirical patch distribution cannot
distinguish the populations. A coordinate-aware joint statistic can.

This is the smallest example of the patch-marginal versus slide-joint gap.

## 8. Mean, Set, And Spatial Fibers

The mean feature is:

```math
P_{\mathrm{mean}}(X)
=
\frac{1}{n}
\sum_{j=1}^{n}
f(x_j).
```

Its fiber contains every bag with the same first moment.

The empirical feature measure is:

```math
P_{\mathrm{set}}(X)
=
\frac{1}{n}
\sum_{j=1}^{n}
\delta_{f(x_j)}.
```

Its fiber contains every coordinate arrangement with the same feature
multiset.

A coordinate-indexed empirical measure is:

```math
P_{\mathrm{spatial}}(X)
=
\frac{1}{n}
\sum_{j=1}^{n}
\delta_{(f(x_j),c_j)}.
```

It refines the feature-only measure:

```math
P_{\mathrm{spatial}}
\succeq
P_{\mathrm{set}}
\succeq
P_{\mathrm{mean}}.
```

The order means each object can deterministically recover the object to its
right. It does not guarantee better finite-sample performance.

## 9. Partition Consistency For Hierarchical Models

Let a slide partition be:

```math
\Pi
=
\{R_1,\ldots,R_m\}.
```

A hierarchical representation is:

```math
z_{\Pi}(X)
=
\mathcal R_{mathrm{slide}}
\left(
    \left\{
        \mathcal R_{\mathrm{region}}
        \left(
            X\vert_{R_r}
        \right)
    \right\}_{r=1}^{m}
\right).
```

Let `Pi_prime` refine `Pi`; every coarse region is a union of fine regions. A
hierarchy is projectively consistent when there exists merge map `M` such that:

```math
z_{\Pi}(X)
=
M
\left(
    z_{\Pi'}(X)
\right).
```

For additive sufficient statistics, this can hold exactly. Let region statistic
be count and sum:

```math
T(R)
=
\left(
    |R|,
    \sum_{j\in R}h_j
\right).
```

For disjoint regions:

```math
T(R_a\cup R_b)
=
T(R_a)
+
T(R_b).
```

A class-token hierarchy is not generally additive. Changing the 4096-region
partition can change HIPT's region tokens even when underlying tissue is fixed.

## 10. Boundary Dependence

Let `B_delta` shift a region grid by offset `delta`. Partition sensitivity is:

```math
\Delta_{\Pi}(\delta)
=
d
\left(
    z_{\Pi}(X),
    z_{B_\delta\Pi}(X)
\right).
```

For a tumor-stroma interface crossing a boundary, two partitions can expose
different within-region interactions before compression.

Boundary dependence is not automatically an error. The partition is part of
the model's geometry. The failure occurs when predictions depend strongly on
an arbitrary origin choice that should be clinically irrelevant.

## 11. Patch-To-Slide Information Monotonicity

Consider the deterministic chain:

```math
X
\longrightarrow
H
\longrightarrow
Z
\longrightarrow
\widehat Y.
```

Data processing gives:

```math
I(Y;\widehat Y)
\le
I(Y;Z)
\le
I(Y;H)
\le
I(Y;X).
```

Every patch encoding and slide readout can only remove task information in the
unrestricted information sense.

Context can improve a restricted downstream reader without violating this
inequality. It can transform distributed information in `H` into an accessible
statistic `Z`, even though it cannot create information absent from `H`.

## 12. Slide-Text Alignment Object

Let a slide and report pair be:

```math
(X_i,T_i)
\sim
P_{XT}.
```

The visual and text encoders produce:

```math
z_i
=
F_X(X_i),
\qquad
t_i
=
F_T(T_i).
```

Contrastive alignment reads cross-modal similarities:

```math
K_{ij}^{XT}
=
z_i^{\mathsf T}t_j.
```

The directly constrained object is a batch-level cross-modal relation matrix,
not a dense correspondence between report phrases and patches.

TITAN aligns ROI captions and WSI reports at different stages. Prov-GigaPath
also explores WSI-report contrastive alignment. CONCH and PLIP align local
images with text. The granularity of `X_i` determines what the pairing can
identify.

## 13. Shared And Private Latent Factors

Let common and modality-private factors be:

```math
C,
\qquad
U_X,
\qquad
U_T.
```

Generate modalities by:

```math
X
=
g_X(C,U_X),
\qquad
T
=
g_T(C,U_T).
```

An aligned representation can preserve mainly the common factor:

```math
z_X
\approx
z_T
\approx
h(C).
```

This can discard visual-private information `U_X` and text-private information
`U_T`.

For visual task `Y_X`, sufficiency requires:

```math
Y_X
\perp
X
\mid
Z_X.
```

Cross-modal alignment requires a different condition:

```math
T
\perp
X
\mid
Z_X.
```

Text sufficiency does not imply visual-task sufficiency.

## 14. Joint Alignment Is Non-Identifiable Up To Shared Symmetry

Suppose the loss depends only on cross-modal dot products:

```math
z_i^{\mathsf T}t_j.
```

For any orthogonal matrix `Q`, transform both modalities:

```math
z_i'
=
Qz_i,
\qquad
t_j'
=
Qt_j.
```

Then:

```math
z_i'^{\mathsf T}t_j'
=
z_i^{\mathsf T}t_j.
```

The aligned coordinate basis is not identifiable. Only the shared relation
geometry is.

If separate invertible transforms are allowed without preserving the metric,
the cross-modal scores change. The symmetry is joint and geometry-dependent.

## 15. One Report Does Not Identify One Patch

Suppose slide representation is a permutation-invariant aggregate:

```math
z_i
=
\mathcal R
\left(
    \{h_{ij}\}_{j=1}^{n_i}
\right),
```

and contrastive loss uses only `(z_i,t_i)`.

For any permutation `pi`:

```math
\mathcal R
\left(
    \{h_{ij}\}_{j=1}^{n_i}
\right)
=
\mathcal R
\left(
    \{h_{i\pi(j)}\}_{j=1}^{n_i}
\right).
```

The slide-report objective is unchanged. It cannot identify which coordinate
or patch supports one phrase without additional structure.

More strongly, two patch-credit vectors can yield the same aggregate:

```math
\sum_{j=1}^{n_i}
a_{ij}h_{ij}
=
\sum_{j=1}^{n_i}
a_{ij}'h_{ij}
```

with:

```math
a_i
\ne
a_i'.
```

The report loss cannot distinguish them when it reads only the aggregate.
Patch localization requires phrase-region pairs, dense annotations,
interventions, architectural constraints, or another identifying signal.

## 16. One-To-Many Pairing And False Negatives

A slide can have several valid reports:

```math
T
\sim
P(T\mid X).
```

Two slides can also share one valid diagnostic description. In-batch contrast
often treats all off-diagonal pairs as negatives:

```math
i\ne j
\Longrightarrow
(X_i,T_j)
\text{ enters negative support}.
```

But:

```math
P(T_j\mid X_i)
```

can still be large. This is a cross-modal false negative caused by one-to-many
semantics.

Knowledge groups in KEEP and prompt ensembles in vision-language evaluation are
different responses to this ambiguity. Neither creates dense visual-text
identifiability by itself.

## 17. Multimodal Fusion

Let modality encoders produce:

```math
Z_i^{(m)}
=
F_m
\left(
    X_i^{(m)}
\right),
\qquad
m=1,\ldots,M.
```

Fusion is:

```math
Z_i^{\mathrm{fuse}}
=
\mathcal F
\left(
    Z_i^{(1)},
    \ldots,
    Z_i^{(M)}
\right).
```

Concatenation preserves each input vector up to the fusion head:

```math
Z^{\mathrm{concat}}
=
[Z^{(1)};\ldots;Z^{(M)}].
```

A bottleneck fusion:

```math
Z^{\mathrm{bottleneck}}
=
W
[Z^{(1)};\ldots;Z^{(M)}]
```

creates null-space collisions. Cross-attention can preserve conditional
interactions but remains limited by the token sets supplied by each modality.

## 18. Missing-Modality Risk

Let missingness mask be:

```math
M
\in
\{0,1\}^{K}.
```

The deployment predictor is:

```math
\widehat Y
=
h
\left(
    \mathcal F
    \left(
        Z^{(1)},\ldots,Z^{(K)};
        M
    \right)
\right).
```

Training only on complete modalities estimates risk under:

```math
P_{\mathrm{train}}(M=\mathbf 1)
=
1.
```

Deployment can satisfy:

```math
P_{\mathrm{deploy}}(M=\mathbf 1)
<
1.
```

The missingness mechanism matters:

```math
P(M\mid X,Y)
```

can depend on disease, site, test ordering, or data availability.

Modality dropout trains over an artificial mask law:

```math
M
\sim
Q_{\mathrm{drop}}.
```

Robustness transfers only when `Q_drop` covers clinically relevant missingness
and the observed modalities contain enough task information.

## 19. Optimal Imputation Does Not Recover Unobserved Realization

For missing modality `T`, the squared-error optimal imputation from visual
representation is:

```math
\widehat T^{\star}
=
\mathbb E[T\mid Z_X].
```

The irreducible error is:

```math
\mathbb E
\left[
    \mathrm{Var}
    \left(
        T
        \mid
        Z_X
    \right)
\right].
```

Generated reports or modality imputations can be plausible conditional means
without reproducing the actual missing clinical observation.

For decision task `Y`, direct prediction from observed modalities can be better
identified than first generating a missing modality and then treating it as
observed truth.

## 20. Retrieval Alignment Is A Different Diagram

Let a query encoder and archive encoder be:

```math
z_q
=
F_q(Q),
\qquad
z_r
=
F_r(X_r).
```

Retrieval is:

```math
\mathcal N_K(Q)
=
\underset{
    A\subseteq\mathcal M,
    |A|=K
}{\arg\min}
\sum_{r\in A}
d(z_q,z_r).
```

Classification is another map:

```math
\widehat Y
=
h(z_q).
```

No commutation is guaranteed between classification and retrieval:

```math
h(z_q)
\not\equiv
\mathrm{Vote}
\left(
    \{y_r:r\in\mathcal N_K(Q)\}
\right).
```

They can agree empirically while using different geometric statistics.

## 21. Object-Alignment Matrix

| Model family | Pretraining source object | Exported object | Downstream object required | Main non-commuting risk |
|---|---|---|---|---|
| CTransPath and RetCCL | augmented patch plus memory or cluster relation | patch vector; RetCCL also constructs retrieval mosaics | WSI set, hierarchy, graph, or retrieval archive | local feature relation does not identify slide layout |
| HIPT | nested patch and 4096-region views | patch and region class-token summaries | task-trained slide composition | fixed partition and repeated class-token compression |
| UNI and Virchow | individual pathology patch views | one vector per patch | explicit WSI reader | patch multiset or mean can collide on layout and interaction |
| PLIP and CONCH | local image-caption pair | shared image-text vector; CONCH also exports caption tokens | top-K, MIL, retrieval, or task head | local text alignment does not identify WSI organization |
| Prov-GigaPath | tile sequence with coordinates and masked slide context | contextual tile set or slide token | downstream ABMIL and task head | serialization, sparse support, and readout prevent exact spatial equivalence |
| TITAN | ROI or WSI feature grid with captions and reports | one-query slide vector and 128-query generation tokens | linear, Cox, zero-shot, retrieval, or decoder reader | one-query and multitoken paths represent different slide objects |
| KEEP | semantic image-caption groups and disease graph | knowledge-conditioned image-text vector | tile voting or tumor-ratio WSI reader | semantic relation does not supply spatial slide context |

## 22. Alignment Audit Protocol

### 22.1 Patch Versus Slide Factorization

Fit transfer map:

```math
\widehat T
=
\arg\min_T
\sum_i
\left\|
    S(X_i)-T(P(X_i))
\right\|_2^2.
```

Evaluate on held-out patients. Compare with a shuffled-slide baseline so
apparent commutation is not caused by cohort or label shortcuts.

### 22.2 Matched-Marginal Counterfactuals

Construct or simulate slides with matched feature distributions and different
coordinates or co-occurrence. Verify which representation changes.

### 22.3 Partition Refinement

Shift and refine region grids while holding pixels fixed. Report representation
and prediction sensitivity.

### 22.4 Phrase-Patch Identifiability

Hold the slide aggregate and report fixed while permuting candidate patch
credits. If the objective is unchanged, localization is not identified by that
objective.

### 22.5 Missing Modality

Evaluate natural and simulated missingness separately. Report performance by
mask pattern and missingness mechanism.

### 22.6 Retrieval Archive

Replace archive composition, remove same-patient cases, and vary metric. A
stable encoder with unstable neighbors identifies a memory-relative failure.

## 23. C/R/G/S Placement

Patch path:

```math
\widetilde H_i^{\mathrm{patch}}
=
\mathcal C_{\mathrm{patch}}
\left(
    H_i;
    G_i,
    S_{\mathrm{pre}}
\right),
```

```math
Z_i^{\mathrm{patch}}
=
\mathcal R_{\mathrm{patch}}
\left(
    \widetilde H_i^{\mathrm{patch}}
\right).
```

Slide path:

```math
Z_i^{\mathrm{slide}}
=
\mathcal R_{\mathrm{slide}}
\left(
    \mathcal C_{\mathrm{slide}}
    \left(
        H_i;
        G_i,
        S_{\mathrm{slide}}
    \right)
\right).
```

Multimodal path:

```math
Z_i^{\mathrm{multi}}
=
\mathcal R_{\mathrm{multi}}
\left(
    \mathcal C_{\mathrm{multi}}
    \left(
        Z_i^{(X)},
        Z_i^{(T)};
        G_{XT},
        S_{XT}
    \right)
\right).
```

The commuting question asks whether one path factors through another. The
surviving-statistic question asks which fibers each `R` creates.

## 24. Bottom Line

Object alignment is a factorization problem:

```math
\boxed{
\text{Does the target representation or task remain constant on every fiber
created by the source representation?}
}
```

Patch pretraining and slide pretraining commute only when the slide object
factors through the patch statistic. Hierarchies commute across partitions only
under a compatible merge law. Slide-report alignment identifies a global
cross-modal relation, not patch localization. Multimodal fusion remains limited
by private information and missingness. Retrieval adds a metric and archive to
the represented object.

The practical question is:

```math
\boxed{
\text{What random object was aligned, what information was already compressed,
and which downstream claim requires a factorization that the training objective
never identified?}
}
```
