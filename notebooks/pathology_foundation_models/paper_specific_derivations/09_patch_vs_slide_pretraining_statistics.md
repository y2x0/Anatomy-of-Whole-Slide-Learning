# Patch Versus Slide Pretraining Statistics

References:

- [HIPT](https://arxiv.org/abs/2206.02647)
- [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w)
- [TITAN](https://www.nature.com/articles/s41591-025-03982-3)
- [pathology foundation-model paper derivations](README.md)

The phrase “foundation model feature” hides a decisive distinction: was the
representation trained on individual patches, on nested regions, or on the
whole slide as one structured object? These objectives expose different
statistics to the encoder.

## 1. Two different random objects

Let patient or slide i contain a patch collection:

```math
\mathcal X_i
=
\left\{
(x_{ij},c_{ij})
\right\}_{j=1}^{n_i},
```

where x-ij is an image patch and c-ij can include coordinates, scale, region,
or other metadata.

A patch-level pretraining sample is:

```math
Z_{\mathrm{patch}}
=
(x_{ij},v_{ij},t_{ij},q_{ij}),
```

where v is an augmentation view, t is optional text, and q is a teacher or
masked target.

A slide-level pretraining sample is:

```math
Z_{\mathrm{slide}}
=
\left(
\{x_{ij},c_{ij}\}_{j=1}^{n_i},
r_i,
t_i
\right),
```

where r-i can be a report and t-i can be a slide-level teacher target.

The marginal patch distribution and joint slide distribution are:

```math
P_{\mathrm{patch}}(x)
=
\sum_i
\frac{n_i}{\sum_k n_k}
P_i(x),
```

```math
P_{\mathrm{slide}}
\left(
\{x_j,c_j\}_{j=1}^{n}
\right).
```

Patch pretraining weights slides in proportion to their sampled patch count
unless sampling is explicitly reweighted. Slide pretraining can weight each
slide or patient once.

## 2. Patch encoder contract

A patch encoder is:

```math
h_{ij}
=
f_\phi(x_{ij}).
```

A patch objective has the form:

```math
\mathcal L_{\mathrm{patch}}
=
\mathbb E_{x,v,v',t,q}
\left[
\ell_{\mathrm{patch}}
\left(
f_\phi(v(x)),
f_\phi(v'(x)),
t,
q
\right)
\right].
```

Examples include:

```text
CTransPath:
    cross-view contrastive patch geometry

RetCCL:
    cluster-guided contrastive retrieval geometry

UNI and Virchow:
    broad pathology patch representation contracts

CONCH and PLIP:
    image-text patch or image-caption alignment

Phikon and masked modeling:
    reconstruction or teacher-target prediction from masked tokens
```

The patch objective directly constrains relations among local views or local
targets. It does not directly constrain relations among two different patches
on the same slide unless those patches enter the objective as an explicit pair.

## 3. Slide encoder contract

A slide encoder maps a structured collection to a slide representation:

```math
z_i
=
F_\psi
\left(
\{h_{ij},c_{ij}\}_{j=1}^{n_i}
\right).
```

A slide-level objective can be:

```math
\mathcal L_{\mathrm{slide}}
=
\mathbb E_i
\left[
\ell_{\mathrm{slide}}
\left(
F_\psi(\mathcal X_i),
t_i
\right)
\right].
```

The map F can preserve:

```text
patch co-occurrence
spatial arrangement
region membership
long-range tissue context
slide-report alignment
slide-level retrieval structure
```

only to the extent that those distinctions affect the objective and survive
the architecture's bottlenecks.

Prov-GigaPath and TITAN are examples of slide-level or slide-aware contracts
whose representation object is not just an unordered collection of independent
patch vectors. HIPT exposes the intermediate case: local and regional
hierarchical objectives create multiscale tokens before downstream slide use.

## 4. Non-equivalence of patch and slide training

A patch encoder followed by a readout is:

```math
z_i^{\mathrm{posthoc}}
=
\mathcal R
\left(
\{f_\phi(x_{ij})\}_{j=1}^{n_i}
\right).
```

A jointly trained slide encoder is:

```math
z_i^{\mathrm{joint}}
=
F_\psi
\left(
\{x_{ij},c_{ij}\}_{j=1}^{n_i}
\right).
```

There is no general identity:

```math
\mathcal R
\left(
\{f_\phi(x_{ij})\}
\right)
\equiv
F_\psi
\left(
\{x_{ij},c_{ij}\}
\right).
```

The first learns a statistic after local compression. The second can use
coordinates, co-occurrence, and cross-patch interactions before the final
representation is formed.

Equality can occur in special cases, for example when F factorizes through
the patch embeddings and R is expressive enough:

```math
F_\psi(\mathcal X_i)
=
\widetilde F_\psi
\left(
\{f_\phi(x_{ij})\}_{j=1}^{n_i}
\right).
```

This is an architectural assumption, not a consequence of using a strong
patch encoder.

## 5. Marginal versus joint information

Let X-1 and X-2 be two patch variables from the same slide. Patch pretraining
can constrain their marginals:

```math
P(h_1),
\qquad
P(h_2).
```

A slide objective can constrain their joint relation:

```math
P(h_1,h_2,c_1,c_2).
```

Two slide distributions can have equal marginals but different joint structure:

```math
P_A(h_1)=P_B(h_1),
\qquad
P_A(h_2)=P_B(h_2),
\qquad
P_A(h_1,h_2)\ne P_B(h_1,h_2).
```

A patch-only statistic can collide on these distributions. A slide encoder with
context access can distinguish them if the joint relation is visible and useful
to S.

This is the mathematical reason that a patch foundation model does not
automatically contain a slide representation.

## 6. A two-slide counterexample

Consider two slides with the same two patch embeddings:

```math
\mathcal X_A=\{a,b\},
\qquad
\mathcal X_B=\{a,b\}.
```

A patch-only encoder produces the same multiset:

```math
\{f(a),f(b)\}_A
=
\{f(a),f(b)\}_B.
```

Now attach different geometry:

```math
c_A(a)=0,\quad c_A(b)=1,
```

```math
c_B(a)=1,\quad c_B(b)=0.
```

A coordinate-aware slide encoder can distinguish the arrangements:

```math
F_\psi(\{f(a),c_A(a),f(b),c_A(b)\})
\ne
F_\psi(\{f(a),c_B(a),f(b),c_B(b)\})
```

while an order-free patch mean must collide.

The distinction is not “better features.” It is access to a different object.

## 7. Sampling-unit weighting

Suppose slide i contributes n-i sampled patches to a patch objective. The
empirical patch risk is:

```math
\widehat{\mathcal L}_{\mathrm{patch}}
=
\frac{1}{\sum_i n_i}
\sum_i\sum_{j=1}^{n_i}
\ell_{ij}.
```

The slide-uniform risk is:

```math
\widehat{\mathcal L}_{\mathrm{slide\text{-}uniform}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\ell_{ij}.
```

They differ by slide weight:

```math
w_i^{\mathrm{patch}}
=
\frac{n_i}{\sum_k n_k},
\qquad
w_i^{\mathrm{uniform}}
=
\frac{1}{N}.
```

If n-i correlates with scanner, tissue area, cancer type, or label, patch
pretraining changes the effective population distribution.

A weighted patch objective can correct this:

```math
\widehat{\mathcal L}_{\mathrm{weighted}}
=
\frac{1}{N}
\sum_i
\frac{1}{n_i}
\sum_j
\ell_{ij}.
```

The weighting convention should be treated as part of S.

## 8. Rare morphology and dilution

Let rare morphology occupy fraction p-i of slide i. Under uniform patch sampling,
the number of sampled examples in a batch of m patches is:

```math
K_i
\sim
\mathrm{Binomial}(m,p_i).
```

The probability of seeing no rare patch is:

```math
\Pr(K_i=0)
=
(1-p_i)^m.
```

For small p-i, this can remain large even when the slide contains clinically
important rare evidence. A patch objective dominated by common morphology can
learn a representation with excellent average loss and weak rare-region
accessibility.

A slide-level objective can increase the weight of rare evidence only if the
slide representation and loss expose it. Mean pooling can still dilute it after
pretraining.

## 9. Patch augmentation contract

Let t and t-prime be transformations used as positive views. Contrastive or
distillation training encourages:

```math
d
\left(
f_\phi(t(x)),
f_\phi(t'(x))
\right)
\text{ small}.
```

This defines an equivalence relation over views:

```math
x\sim_{\mathrm{pre}}x'
\quad
\text{if the objective treats them as positive views}.
```

The contract is appropriate only if the transformations preserve the
downstream-relevant pathology. If a transformation changes lesion morphology,
then the objective may erase useful information.

The downstream representation must satisfy a different desired relation:

```math
x\sim_{\mathrm{task}}x'
\quad
\text{if they should share the task-relevant state}.
```

A mismatch between the two equivalence relations is a pretraining failure:

```math
\sim_{\mathrm{pre}}
\ne
\sim_{\mathrm{task}}.
```

## 10. Patch text and slide text are different targets

For patch-text alignment:

```math
u_{ij}
=
f_{\mathrm{img}}(x_{ij}),
\qquad
v_{ij}
=
f_{\mathrm{text}}(t_{ij}).
```

For slide-report alignment:

```math
u_i
=
F_{\mathrm{slide}}
\left(
\{x_{ij},c_{ij}\}_j
\right),
\qquad
v_i
=
f_{\mathrm{text}}(r_i).
```

Patch text often describes local morphology. A report can describe a diagnosis,
global extent, or treatment context. The conditional distributions differ:

```math
P(t_{ij}\mid x_{ij})
\ne
P(r_i\mid\mathcal X_i).
```

A model trained on report alignment may preserve global diagnosis language
while ignoring rare local evidence that is absent from the report. A patch
model may preserve local morphology while lacking slide-level diagnostic
composition.

## 11. Hierarchical pretraining as an intermediate object

HIPT constructs nested tokens:

```math
H^{(0)}
\longrightarrow
H^{(1)}
\longrightarrow
H^{(2)}
\longrightarrow
z.
```

A local [CLS] summary is:

```math
h_r
=
\mathrm{CLS}
\left(
\mathcal C_{\mathrm{local}}(H_r^{(0)})
\right).
```

A regional summary is:

```math
z_i
=
\mathrm{CLS}
\left(
\mathcal C_{\mathrm{regional}}(\{h_r\}_r)
\right).
```

This preserves some cross-patch context before the slide-level representation,
but each [CLS] map is a bottleneck:

```math
H_r^{(0)}\ne H_r^{(0)\prime},
\qquad
h_r=h_r'.
```

A later stage cannot recover the lost child contrast. Hierarchical pretraining
is therefore neither equivalent to independent patch pretraining nor to a
lossless slide encoder.

## 12. Slide objective and co-occurrence

Suppose a slide label depends on a co-occurrence relation:

```math
Y
=
\mathbf 1
\left\{
\exists j,k:
x_j\in A,\ x_k\in B,\ d(c_j,c_k)\le r
\right\}.
```

A patch encoder learns f(x-j) independently. A posthoc mean cannot generally
represent the existence of the spatially close A-B pair:

```math
\frac{1}{n}\sum_jf(x_j)
```

does not retain which features belonged to neighboring coordinates.

A slide encoder with coordinate context can use:

```math
F_\psi
\left(
\{f(x_j),c_j\}_j
\right)
```

to preserve the pair relation, subject to its context and readout capacity.

## 13. Patch versus slide objective gradients

For patch pretraining:

```math
\nabla_\phi\mathcal L_{\mathrm{patch}}
=
\sum_{i,j}
\frac{\partial\ell_{ij}}{\partial f_\phi(x_{ij})}
\frac{\partial f_\phi(x_{ij})}{\partial\phi}.
```

For a slide objective:

```math
\nabla_\psi\mathcal L_{\mathrm{slide}}
=
\sum_i
\frac{\partial\ell_i}{\partial z_i}
\frac{\partial z_i}{\partial H_i}
\frac{\partial H_i}{\partial\psi}.
```

The second gradient contains cross-patch terms whenever F is contextual:

```math
\frac{\partial z_i}{\partial h_{ij}}
\ne
\frac{\partial z_i}{\partial h_{ik}}
\quad
\text{only because }j\ne k;
```

and more generally:

```math
\frac{\partial h_{ik}^{\mathrm{context}}}{\partial h_{ij}}
\ne0
\quad
\text{for }j\ne k.
```

Slide training can therefore reshape local features according to their
relational role, not only their isolated appearance.

## 14. Readout-limited transfer

A pretrained slide representation z can contain a task signal that a readout
does not expose. Let the frozen representation be:

```math
z_i
=
F_\psi(\mathcal X_i).
```

A downstream head is restricted to:

```math
\widehat y_i
=
\mathcal H_\eta
\left(
\mathcal R_\eta(z_i)
\right).
```

If Y is not predictable in the chosen class H-eta composed with R-eta, low
performance does not prove the signal was absent from z.

For patch features:

```math
\widehat y_i
=
\mathcal H_\eta
\left(
\mathcal R_\eta(\{f_\phi(x_{ij})\})
\right).
```

A stronger MIL readout can expose information that a mean probe misses. The
encoder and readout should be evaluated separately.

## 15. Representation equivalence relation

For a complete slide learner, define:

```math
S\sim_F S'
\quad\Longleftrightarrow\quad
F(S)=F(S').
```

The pretraining objective induces a desired equivalence relation over views,
patches, slides, reports, or augmentations. A downstream task induces another:

```math
S\sim_Y S'
\quad\Longleftrightarrow\quad
P(Y\mid S)=P(Y\mid S').
```

A foundation representation is sufficient for a task only if its collisions
mostly occur within task-equivalent classes:

```math
F(S)=F(S')
\Longrightarrow
S\sim_Y S'
```

for the relevant support. Violations are missing-statistic failures.

This is a sufficiency condition, not a claim that a finite representation can
be injective on all slides.

## 16. Distribution shift in patch and slide geometry

Let pretraining and deployment distributions be P-pre and P-deploy. Patch
geometry can remain stable while slide geometry shifts:

```math
P_{\mathrm{pre}}(h)
\approx
P_{\mathrm{deploy}}(h),
\qquad
P_{\mathrm{pre}}(c\mid h)
\ne
P_{\mathrm{deploy}}(c\mid h).
```

Examples include:

```text
different tessellation or tile stride
different coordinate normalization
missing tissue or slide cropping
scanner and stain shift
different report template or institution language
different region count and slide area
```

A slide encoder that uses c or report context can change even when local patch
embeddings appear stable.

## 17. Sample complexity and slide imbalance

Let N be the number of slides and n-i the number of patches per slide. A
patch-level objective may have many training terms but only N independent
patient-level units. Treating all patches as independent can understate variance.

A slide-uniform estimator is:

```math
\widehat\theta
=
\frac{1}{N}
\sum_i
\frac{1}{n_i}
\sum_j
\ell_{ij}.
```

Its variance depends on within-slide correlation:

```math
\mathrm{Var}
\left(
\frac{1}{n_i}\sum_j\ell_{ij}
\right)
=
\frac{1}{n_i^2}
\sum_{j,k}
\mathrm{Cov}(\ell_{ij},\ell_{ik}).
```

Large n-i does not imply n-i independent examples. Tissue patches from one
patient can be highly correlated.

A slide-level objective has fewer terms but can align the unit of optimization
with the unit of evaluation. The effective sample size should be discussed at
the patient level.

## 18. Patch-to-slide transfer matrix

| Pretraining object | Context before objective | Statistic directly constrained | Typical downstream loss |
| --- | --- | --- | --- |
| patch views | local augmentation orbit | local invariance and pair geometry | MIL, linear probe |
| patch-text pairs | local image-language relation | caption-visible semantics | retrieval, prompt classification |
| nested regions | local-to-regional token context | multiscale summaries | slide classification, survival |
| full slide | cross-patch and coordinate context | slide joint statistic | slide task or report alignment |
| slide-report pair | global visual-language relation | diagnosis/report-visible statistic | retrieval, zero-shot, adaptation |
| retrieval memory | cohort neighborhood | metric-relative neighbor structure | kNN, example-based transfer |

The object, not the model size, determines what the objective can directly
constrain.

## 19. Paper-specific interpretation

### HIPT

HIPT's hierarchical DINO training constrains local and regional teacher-student
relations. Its [CLS] tokens are learned summaries at each spatial scale. The
main transfer limit is the information discarded at each [CLS] boundary and
the dependence on fixed window geometry.

### Prov-GigaPath

Prov-GigaPath separates tile-level foundation features from a slide-level
long-context model. The slide representation can preserve cross-tile context
that a frozen tile mean cannot. Its limits include coordinate/tessellation
dependence, missing-tile sensitivity, and the finite slide-level bottleneck.

### TITAN

TITAN's slide and text representation contract exposes global semantic and
retrieval geometry. Its transfer can be strong for report-visible concepts
while remaining sensitive to report distribution, retrieval archive, and the
chosen slide readout.

These are distinct claims and should not be collapsed into “patch versus
slide.”

## 20. C/R/G/S placement

```math
\begin{aligned}
G &: \text{views, coordinates, hierarchy, slide order, and modality pairing},
\\
C &: \text{patch encoder, regional context, slide encoder, or text fusion},
\\
R &: \text{patch pooling, [CLS], attention, retrieval, or task head},
\\
S &: \text{contrastive, distillation, masked, report, or slide-level loss}.
\end{aligned}
```

A patch foundation model can have strong C at the patch scale but no cross-patch
C. A slide foundation model can have cross-patch C but still lose information
at R.

## 21. Audit protocol

### 21.1 Sampling audit

Compare patch-weighted and slide-uniform objectives:

```math
\widehat{\mathcal L}_{\mathrm{patch}}
\quad
\text{versus}
\quad
\widehat{\mathcal L}_{\mathrm{slide\text{-}uniform}}.
```

### 21.2 Co-occurrence audit

Construct slide pairs with matched patch marginals and different spatial
arrangements. Measure whether z changes.

### 21.3 Readout audit

Compare mean, attention, prototype, and hierarchy readouts on the same frozen
patch features.

### 21.4 Scale audit

Remove or perturb one region at a time and measure the slide representation
change:

```math
\Delta_r
=
d
\left(
F(\mathcal X),
F(\mathcal X\setminus r)
\right).
```

### 21.5 External shift audit

Evaluate by patient, institution, stain, scanner, slide area, and rare morphology
strata. Patch and slide models can fail in different strata.

## 22. Bottom line

Patch-level and slide-level foundation models optimize different statistical
contracts:

```math
\boxed{
\begin{aligned}
\text{patch training}
&\longrightarrow
\text{local view and target geometry},
\\
\text{slide training}
&\longrightarrow
\text{joint, spatial, regional, or multimodal slide geometry}.
\end{aligned}
}
```

A strong patch encoder is a powerful input representation, not a proof that a
posthoc readout can recover every slide statistic. A strong slide encoder is not
a proof of local feature sufficiency or external robustness.

The correct question is:

```math
\boxed{
\text{what random object did pretraining see, what equivalence relation did it
reward, and which statistic remains accessible to the downstream reader?}
}
```
