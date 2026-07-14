# Invariance And Surviving Statistics

Primary anchors:

- [DINOv2](https://arxiv.org/abs/2304.07193)
- [Masked Autoencoders](https://arxiv.org/abs/2111.06377)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [HIPT](https://arxiv.org/abs/2206.02647)
- [UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/)
- [Virchow](https://arxiv.org/abs/2309.07778)
- [CONCH](https://arxiv.org/abs/2307.12914)
- [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w)
- [TITAN](https://www.nature.com/articles/s41591-025-03982-3)
- [KEEP](https://arxiv.org/html/2412.13126v1)
- [pretraining contract matrix](01_pretraining_contract_matrix.md)

Pretraining does not preserve "pathology" as one undifferentiated object. It
creates pressure to identify some inputs, separate others, predict selected
targets, and compress the result through an exported statistic.

The central question is:

```math
\text{Which distinctions become invisible in the representation, and which
measurable statistic remains available after that identification?}
```

## 1. Exact Invariance Under A Group Action

Let a transformation group `Gamma` act on input space `X`:

```math
\Gamma
\times
\mathcal X
\longrightarrow
\mathcal X,
\qquad
(g,x)
\longmapsto
g\cdot x.
```

The orbit of `x` is:

```math
[x]_{\Gamma}
=
\left\{
    g\cdot x
    :
    g\in\Gamma
\right\}.
```

A representation is invariant when:

```math
F(g\cdot x)
=
F(x)
\qquad
\forall g\in\Gamma.
```

Let the quotient projection be:

```math
\pi_{\Gamma}
:
\mathcal X
\longrightarrow
\mathcal X/\Gamma,
\qquad
\pi_{\Gamma}(x)
=
[x]_{\Gamma}.
```

Every invariant representation factors through the quotient:

```math
F
=
\widetilde F
\circ
\pi_{\Gamma}
```

for some map `F_tilde` on the orbit space.

The quotient is the mathematical statement of what the representation cannot
see. Once `x` and `g dot x` share one orbit, an invariant downstream system
cannot distinguish them.

## 2. Maximal Invariant

An invariant statistic `T` is maximal when:

```math
T(x)
=
T(x')
\quad
\Longleftrightarrow
\quad
x'
\in
[x]_{\Gamma}.
```

A maximal invariant removes exactly the group nuisance and no additional
distinction. A non-maximal invariant can merge several orbits:

```math
T(x)
=
T(x')
\quad
\text{while}
\quad
[x]_{\Gamma}
\ne
[x']_{\Gamma}.
```

That extra collision may be harmless for one task and fatal for another.

Foundation-model invariance claims should therefore separate:

```text
desired orbit collapse;
additional architecture or readout collapse;
finite-data failure to distinguish two orbits;
task-specific safety of the resulting collision.
```

## 3. Task Compatibility Of An Invariance

Let the task conditional law be:

```math
P(Y\mid X=x).
```

The transformation group is task-compatible when:

```math
P(Y\mid X=x)
=
P(Y\mid X=g\cdot x)
```

for every relevant `g` and almost every `x`.

Equivalently, the orbit variable is task-sufficient when:

```math
Y
\perp
X
\mid
\pi_{\Gamma}(X).
```

If this condition fails, no exactly invariant representation can be sufficient
for that task.

Under logarithmic loss, raw-input Bayes risk is:

```math
\mathcal R_{\log}^{X}
=
H(Y\mid X).
```

For any invariant representation `Z`, there exists `h` such that:

```math
Z
=
h
\left(
    \pi_{\Gamma}(X)
\right).
```

Therefore:

```math
H(Y\mid Z)
\ge
H
\left(
    Y
    \mid
    \pi_{\Gamma}(X)
\right).
```

The unavoidable log-loss gap obeys:

```math
\boxed{
\mathcal R_{\log}^{Z}
-
\mathcal R_{\log}^{X}
\ge
I
\left(
    Y;
    X
    \mid
    \pi_{\Gamma}(X)
\right).
}
```

Equality is possible when `Z` is itself sufficient for the orbit variable and
does not create additional task-relevant collisions.

This is the exact cost of declaring a task-relevant transformation to be a
nuisance.

## 4. Stochastic Augmentations Are Usually Not A Group

Pathology view construction is often a Markov kernel:

```math
T
\sim
K(\cdot\mid x),
\qquad
V=T(x).
```

Random crop, masking, tissue filtering, and color jitter need not be invertible
or closed under composition. The group quotient is then only an analogy.

Define co-support relation:

```math
v
\mathrel{R_K}
v'
\quad
\Longleftrightarrow
\quad
\exists x
\text{ such that }
K(v\mid x)>0
\text{ and }
K(v'\mid x)>0.
```

Let:

```math
\sim_K
=
\mathrm{EqCl}(R_K)
```

be its equivalence closure. Repeated positive alignment can create pressure
toward constancy along connected components of this relation graph.

The closure can be much larger than one augmentation pair. A chain of locally
plausible positives can connect endpoints that differ in clinically important
ways.

## 5. Positive Alignment As Graph-Laplacian Smoothing

Let sampled views be graph nodes:

```math
\mathcal V_B
=
\{v_1,\ldots,v_B\}.
```

Let symmetric positive weights be:

```math
W_{ab}
\ge
0.
```

Define degree and graph Laplacian:

```math
D_{aa}
=
\sum_{b=1}^{B}W_{ab},
\qquad
L
=
D-W.
```

For representation matrix:

```math
Z
=
\begin{bmatrix}
z_1^{\mathsf T}\\
\vdots\\
z_B^{\mathsf T}
\end{bmatrix},
```

the squared alignment energy is:

```math
\mathcal E_{+}(Z)
=
\frac{1}{2}
\sum_{a,b=1}^{B}
W_{ab}
\|z_a-z_b\|_2^2
=
\mathrm{tr}
\left(
    Z^{\mathsf T}LZ
\right).
```

If the positive graph is connected:

```math
\mathcal E_{+}(Z)
=
0
\Longrightarrow
z_1
=
\cdots
=
z_B.
```

Positive alignment alone therefore admits complete collapse. Contrastive
negatives, centering, sharpening, variance constraints, teacher asymmetry, or
other regularizers are needed to prevent the constant solution.

## 6. Effective Resistance Bounds Approximate Invariance

For two nodes in the same positive graph component, let effective resistance be:

```math
R_{\mathrm{eff}}(a,b)
=
(e_a-e_b)^{\mathsf T}
L^{\dagger}
(e_a-e_b).
```

The representation difference is bounded by graph energy:

```math
\boxed{
\|z_a-z_b\|_2^2
\le
R_{\mathrm{eff}}(a,b)
\mathcal E_{+}(Z).
}
```

Thus even pairs never observed directly can become close when many low-
resistance positive paths connect them.

This is useful for memory-mined and cluster-expanded positives. A single false
edge can create a short path between clinically distinct components and spread
alignment pressure beyond that one pair.

## 7. Repulsion Changes The Quotient Geometry

A contrastive objective adds negative support:

```math
\mathcal E_{-}(Z)
=
\sum_{a,b}
A_{ab}^{-}
\ell_{-}(z_a,z_b).
```

The learned geometry balances:

```math
\mathcal J(Z)
=
\mathcal E_{+}(Z)
+
\lambda
\mathcal E_{-}(Z)
+
\Omega(Z).
```

False positives merge task-distinct points. False negatives repel
task-equivalent points. Their effects are not symmetric:

```text
false positive:
    directly enlarges a learned equivalence component;

false negative:
    fragments a semantic class or distorts its metric diameter.
```

CTransPath changes positive support through memory mining. RetCCL changes
negative weighting and centroid targets. The surviving geometry depends on the
whole relation graph.

## 8. Invariance Versus Equivariance

Invariance removes transformation state:

```math
F(g\cdot x)
=
F(x).
```

Equivariance preserves it in a structured representation:

```math
F(g\cdot x)
=
\rho(g)F(x),
```

where `rho` is a representation of the transformation group.

If a downstream readout is invariant to `rho`:

```math
\mathcal R(\rho(g)z)
=
\mathcal R(z),
```

then the full prediction is invariant while intermediate features retain
transformation information.

For spatial pathology, this distinction matters:

```text
coordinate permutation invariant set model:
    discards layout;

translation-equivariant grid or graph model:
    preserves relative layout before pooling;

coordinate-aware slide transformer:
    can preserve and use absolute or relative position;

augmentation-invariant patch encoder:
    can ignore crop or color state inside each local field.
```

Calling all four "robust" hides different surviving objects.

## 9. Approximate Invariance And Downstream Stability

Suppose:

```math
d_Z
\left(
    F(x),
    F(Tx)
\right)
\le
\varepsilon.
```

If downstream head `h` is `L_h`-Lipschitz:

```math
d_Y
\left(
    h(F(x)),
    h(F(Tx))
\right)
\le
L_h\varepsilon.
```

Approximate representation invariance gives prediction stability only after
accounting for head sensitivity.

Conversely, stable predictions do not prove stable representations. A constant
or saturated head can satisfy:

```math
h(F(x))
\approx
h(F(Tx))
```

while the representations differ substantially.

Audit both levels:

```math
\Delta_Z(T)
=
d_Z(F(x),F(Tx)),
```

```math
\Delta_Y(T)
=
d_Y(h(F(x)),h(F(Tx))).
```

## 10. Surviving Statistic As A Sigma-Field

For deterministic representation:

```math
Z
=
F(X),
```

the information available downstream is the sigma-field:

```math
\sigma(Z)
\subseteq
\sigma(X).
```

Every downstream prediction based only on `Z` is measurable with respect to
`sigma(Z)`.

The phrase "surviving statistic" refers to this measurable object, not only to
the vector dimension. Two same-dimensional vectors can generate different
sigma-fields. A large vector can still be constant on a task-relevant factor.

Task-retained information is:

```math
I(Y;Z).
```

Information discarded by the representation is:

```math
I(Y;X\mid Z).
```

The objective does not generally maximize `I(Y;Z)` because downstream `Y` is
unknown during pretraining.

## 11. Contrastive Surviving Statistic

For normalized embeddings, contrastive training directly reads pairwise inner
products:

```math
K_{ab}
=
z_a^{\mathsf T}z_b.
```

The directly constrained batch object is a similarity matrix:

```math
K
=
ZZ^{\mathsf T}.
```

An orthogonal change of coordinates:

```math
Z'
=
ZQ,
\qquad
Q^{\mathsf T}Q=I,
```

preserves the matrix:

```math
Z'Z'^{\mathsf T}
=
ZZ^{\mathsf T}.
```

The contrastive objective therefore organizes relational geometry, not an
absolute semantic basis. CTransPath and RetCCL differ because they modify the
support and weights entering this geometry.

## 12. Teacher-Student Surviving Statistic

Let teacher target be:

```math
Q_t
=
q_{\bar\theta}(V_t).
```

If the student family can match it and the divergence is strictly proper, the
population optimum satisfies:

```math
p_{\theta}^{\star}(V_s)
=
Q_t
```

on the observed support, after conditioning on the student view.

The teacher target is itself a compressed random variable:

```math
\sigma(Q_t)
\subseteq
\sigma(V_t).
```

Student agreement does not recover distinctions absent from `Q_t`. Extra
student feature directions can survive, but the distillation term does not
identify them.

HIPT, UNI, and Virchow inherit this teacher-defined geometry through DINO-family
training.

## 13. Masked-Modeling Surviving Statistic

Partition tokens into observed and masked sets:

```math
X
=
(X_O,X_M).
```

Under squared reconstruction, the Bayes predictor is:

```math
\widehat X_M^{\star}
=
\mathbb E[X_M\mid X_O,M].
```

The irreducible reconstruction error is:

```math
\mathcal R_{\mathrm{mask}}^{\star}
=
\mathbb E
\left[
    \mathrm{Var}
    \left(
        X_M
        \mid
        X_O,M
    \right)
\right]
```

for scalar targets, with trace covariance for vector targets.

The objective rewards conditional predictability. If a rare event is
independent of visible context:

```math
X_M^{\mathrm{rare}}
\perp
X_O,
```

its instance-specific realization cannot be reconstructed from the context.
The optimal predictor returns its conditional mean or distribution.

Prov-GigaPath's slide masked objective predicts hidden tile embeddings from
visible slide context. Phikon-style masked modeling predicts hidden patch-token
targets from local visible context. They preserve different conditional laws.

## 14. Image-Text Surviving Statistic

Let report or caption be generated through a reporting channel:

```math
T
\sim
P(T\mid X).
```

Image-text alignment rewards information shared with that channel. It does not
require preservation of image information conditionally absent from text:

```math
I(X;Y\mid T)
```

can be positive even when image-text alignment is optimal for the observed pair
distribution.

For a minimal text-sufficient visual statistic `Z_T`:

```math
T
\perp
X
\mid
Z_T.
```

A downstream visual task is safe only when:

```math
Y
\perp
X
\mid
Z_T.
```

The first condition does not imply the second.

PLIP and CONCH preserve caption-visible geometry. TITAN adds ROI and WSI text
channels. KEEP changes the reporting prior through knowledge-conditioned
groups. Each channel makes different visual distinctions visible to the loss.

## 15. Multimodal Common And Private Information

Let visual and textual modalities be generated from latent factors:

```math
X
=
g_X(C,U_X),
\qquad
T
=
g_T(C,U_T),
```

where `C` is common information and `U_X`, `U_T` are modality-private factors.

Pure alignment can be satisfied by preserving mostly `C`:

```math
Z_X
\approx
Z_T
\approx
f(C).
```

Visual-only task information in `U_X` can disappear from the aligned vector.
A generative image token set may retain more private visual information than a
single contrastive vector:

```math
\sigma(Z_{\mathrm{contrastive}})
\subseteq
\sigma(Z_{\mathrm{generative}})
```

only if the first is a measurable function of the second. Different query
poolers do not guarantee this factorization, so in general the two objects are
incomparable.

CONCH's one-query and 256-query objects and TITAN's one-query and 128-query
objects must therefore be audited separately.

## 16. Patch Marginals Versus Slide Joint Structure

Let two slides have patch variables and coordinates:

```math
\mathcal X_i
=
\{(X_{ij},C_{ij})\}_{j=1}^{n_i}.
```

Patch pretraining can organize the marginal feature law:

```math
P(H),
\qquad
H=f(X).
```

Slide pretraining can access a joint law:

```math
P
\left(
    H_1,\ldots,H_n,
    C_1,\ldots,C_n
\right).
```

Two slide populations can satisfy:

```math
P_A(H)
=
P_B(H)
```

while:

```math
P_A(H_1,H_2,C_1,C_2)
\ne
P_B(H_1,H_2,C_1,C_2).
```

Every method using only the empirical patch distribution collides on matched
marginals. A contextual slide encoder can distinguish the populations only if
its interaction support and readout preserve the differing joint factor.

HIPT, Prov-GigaPath, and TITAN expose different forms of joint slide context.
UNI and Virchow supply patch geometry that still needs a slide operator.

## 17. Retrieval Surviving Statistic

For fixed embedding, metric, and archive:

```math
\mathcal N_K(z)
=
\underset{
    A\subseteq\mathcal M,
    |A|=K
}{\arg\min}
\sum_{a\in A}
d(z,z_a).
```

The surviving retrieval statistic is an archive-relative neighborhood. It is
not only the vector `z`:

```math
\mathcal N_K
=
\mathcal N_K
\left(
    z,
    d,
    \mathcal M,
    K
\right).
```

An invertible anisotropic transform can preserve all vector information while
changing Euclidean neighbors. Replacing the archive can change neighbors while
keeping both encoder and metric fixed.

RetCCL uses diagnosis-aware archive retrieval. TITAN uses slide and cross-modal
retrieval from a one-query vector. KEEP and CONCH support text-defined nearest
directions. The memory and metric are part of the surviving decision object.

## 18. A Rare-Lesion Quotient Counterexample

Let slide patch state be:

```math
X
=
(B,L),
```

where `B` is common background morphology and `L` indicates one rare lesion.
Let augmentation `T_drop` delete the lesion with positive probability.

If the alignment contract reaches exact invariance:

```math
F(B,L=1)
=
F
\left(
    T_{\mathrm{drop}}(B,L=1)
\right)
=
F(B,L=0).
```

For downstream label:

```math
Y=L,
```

the representation has:

```math
I(Y;F(X))
=
0
```

under balanced `L` and complete collision.

Increasing model size cannot repair a distinction the exact contract identifies
away. The repair must change the view law, target relation, auxiliary objective,
or exported object.

## 19. Approximate Collision And Margin

Suppose two task classes have representation sets `Z_0` and `Z_1`. Define the
interclass margin:

```math
\gamma
=
\inf_{z_0\in\mathcal Z_0,
      z_1\in\mathcal Z_1}
d_Z(z_0,z_1).
```

If augmentation perturbation can move a class-1 point by at least the margin:

```math
d_Z(F(x),F(Tx))
\ge
\gamma,
```

the transformation can cross the class geometry even without exact collision.

Conversely, a nuisance perturbation bounded by `epsilon` is safe for a
classifier with score margin larger than its induced logit change. For linear
score `w`:

```math
\left|
    w^{\mathsf T}F(x)
    -
    w^{\mathsf T}F(Tx)
\right|
\le
\|w\|_2
\|F(x)-F(Tx)\|_2.
```

This connects representation invariance to downstream robustness without
pretending that one embedding distance has universal clinical meaning.

## 20. Finite-Sample Invariance Estimation

For patient `i`, sample one paired transformation and define:

```math
D_i
=
d_Z
\left(
    F(X_i),
    F(T_iX_i)
\right).
```

The patient-level estimator is:

```math
\widehat\Delta_T
=
\frac{1}{N}
\sum_{i=1}^{N}D_i.
```

Its standard error is:

```math
\widehat{\mathrm{SE}}
\left(
    \widehat\Delta_T
\right)
=
\sqrt{
    \frac{
        \widehat{\mathrm{Var}}(D_i)
    }{N}
}.
```

Sampling many correlated patches from one patient refines the within-patient
estimate but does not create independent biological replicates.

For equivalence, choose practical margin `epsilon` and require the confidence
interval to lie inside:

```math
[-\varepsilon,+\varepsilon].
```

Failure to reject a difference is not evidence of invariance.

## 21. Objective-To-Statistic Matrix

| Objective family | Relation directly read by the loss | Candidate invariance or prediction | Directly surviving object | Main collision risk |
|---|---|---|---|---|
| instance contrast | same-source views versus sampled negatives | invariance along positive view graph | normalized pairwise similarity geometry | lesion-altering views or false positives merge task states |
| cluster or memory contrast | generated neighborhood and centroid relations | stability within mined or clustered support | archive-relative angular and partition geometry | site, stain, or early cluster error defines support |
| self-distillation | teacher distribution across views or masks | teacher-student agreement | teacher-defined output and extra unconstrained backbone state | teacher collapse or teacher-invariant task signal |
| masked modeling | hidden target conditional on visible context | predictability from context | conditional mean, distribution, or token target | rare unpredictable morphology is averaged away |
| image-text contrast | paired image-caption relation | cross-modal semantic alignment | common cosine geometry | report-invisible visual signal or prompt shortcut |
| image-conditioned generation | next token given image token set and prefix | text-predictive visual state | conditional generation token set | templated language or reporting channel dominates |
| slide context pretraining | cross-tile support and slide target | joint spatial or co-occurrence structure | contextual tile or slide statistic | sparse support, coordinate, mask, or readout bottleneck |
| knowledge-conditioned alignment | disease attributes, groups, and graph relations | hierarchy-consistent semantic grouping | knowledge-biased image-text geometry | graph error becomes an alignment prior |
| distillation | teacher output, feature, or relation | student agreement with selected teacher object | compressed teacher statistic | teacher blind spot or unpreserved relation |

The table names direct objective pressure. Extra information can survive, but
its survival is not certified by the objective alone.

## 22. Paper Placement

### CTransPath And RetCCL

Their core invariance is patch-view stability. Their distinctive surviving
statistics come from memory-mined or cluster-shaped angular neighborhoods. WSI
layout remains outside the patch encoder.

### HIPT

Local and region teacher-student invariances are composed through fixed
hierarchical class-token bottlenecks. Within-region position can survive;
absolute WSI arrangement is weaker at the reported slide input.

### UNI And Virchow

Both learn broad DINOv2-style patch geometry. UNI exports the original model's
class token. Virchow exports a class token concatenated with a first moment of
local tokens. Neither export is itself a WSI spatial relation.

### CONCH And PLIP

Their normalized vectors preserve image-text relation under the training pair
channel. CONCH additionally exports captioning tokens. MI-Zero WSI transfer
preserves class-specific upper order statistics over independent tile scores.

### Prov-GigaPath

Tile DINOv2 invariance is followed by conditional prediction of masked tile
embeddings from coordinate-indexed LongNet context. The slide statistic can
preserve joint context absent from a patch marginal, subject to sparse support
and readout.

### TITAN

Vision-only grid context, ROI-caption alignment, and WSI-report alignment impose
different invariances at different scales. One-query retrieval and 128-query
generation expose different surviving objects.

### KEEP

Disease attributes, semantic groups, and hypernym-aware negatives define the
semantic equivalence prior. The WSI tumor-ratio reader then preserves hard tile
class frequency rather than spatial organization.

## 23. C/R/G/S Placement

```math
\widetilde H
=
\mathcal C
\left(
    H;
    G,
    S
\right),
\qquad
U
=
\mathcal R
\left(
    \widetilde H
\right),
\qquad
\widehat Q
=
\mathcal H(U).
```

The invariance analysis assigns:

```text
C:
    where relations propagate and which positive paths can smooth features;

R:
    which token, moment, order statistic, or neighborhood survives;

G:
    group orbit, stochastic view graph, spatial support, metric, or modality
    geometry;

S:
    positive, negative, teacher, mask, caption, report, or knowledge target;

H:
    projection or prediction head through which the objective sees U.
```

The objective constrains `H(U)`. It does not guarantee that `U` is maximal,
task-sufficient, or uniquely identified.

## 24. Bottom Line

Invariance is a quotient operation:

```math
\boxed{
\text{input distinctions}
\longrightarrow
\text{view or relation graph}
\longrightarrow
\text{identified components}
\longrightarrow
\text{exported surviving statistic}.
}
```

The right question is not whether a model is invariant. Every useful model
must ignore something.

The right question is:

```math
\boxed{
\text{Is the learned quotient compatible with the downstream task, does the
exported statistic separate the remaining orbits, and which extra collisions
are introduced by context, readout, metric, or finite data?}
}
```
