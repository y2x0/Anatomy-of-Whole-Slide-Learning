# Paper-Specific Foundation Model Failure Matrix

Primary sources:

- [CTransPath](https://pubmed.ncbi.nlm.nih.gov/35952419/)
- [RetCCL](https://pubmed.ncbi.nlm.nih.gov/36270093/)
- [HIPT](https://arxiv.org/abs/2206.02647)
- [UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/)
- [Virchow](https://arxiv.org/abs/2309.07778)
- [PLIP](https://pubmed.ncbi.nlm.nih.gov/37592105/)
- [CONCH](https://arxiv.org/abs/2307.12914)
- [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w)
- [TITAN](https://www.nature.com/articles/s41591-025-03982-3)
- [KEEP](https://arxiv.org/html/2412.13126v1)
- [Unified Knowledge Distillation](https://arxiv.org/html/2407.18449v3)
- [paper-specific derivations in this folder](README.md)

A useful failure matrix is not a list of vague risks. It is a set of
falsifiable contracts:

```text
which mathematical object enters the model;
which relation pretraining is intended to preserve;
which perturbation should leave that relation unchanged;
which perturbation should change it;
which exported statistic is measured;
which alternative mechanism can mimic the expected result.
```

The audit must be paper-specific because CTransPath mines a memory
neighborhood, HIPT compresses a fixed hierarchy, Prov-GigaPath reconstructs
masked slide tokens, TITAN exports two different slide readouts, and KEEP
changes the text-side prior with a disease graph. One generic "domain shift"
row does not test the same mechanism in all five systems.

## 1. A Two-Sided Representation Contract

For model family `b`, write its exported representation as:

```math
Z_b
=
F_b(X).
```

Let `A_b^0` be a family of transformations declared irrelevant by the
pretraining contract. Examples include crop or stain views intended to preserve
morphology. Let `A_b^1` be a family of interventions intended to change a
diagnostic, spatial, semantic, or retrieval factor.

The desired two-sided behavior is:

```math
\begin{aligned}
a_0\in\mathcal A_b^0
&\Longrightarrow
d_b
\left(
    F_b(X),
    F_b(a_0X)
\right)
\text{ is small},
\\
a_1\in\mathcal A_b^1
&\Longrightarrow
d_b
\left(
    F_b(X),
    F_b(a_1X)
\right)
\text{ is large enough to expose the change}.
\end{aligned}
```

The invariance error is:

```math
\mathcal E_{b}^{\mathrm{inv}}
=
\mathbb E_{X,a_0}
\left[
    d_b
    \left(
        F_b(X),
        F_b(a_0X)
    \right)
\right].
```

For a desired sensitivity margin `m_b`, define:

```math
\mathcal E_{b}^{\mathrm{sens}}
=
\mathbb E_{X,a_1}
\left[
    \left[
        m_b
        -
        d_b
        \left(
            F_b(X),
            F_b(a_1X)
        \right)
    \right]_+
\right].
```

Low invariance error alone is not evidence of a useful model. A constant map
has perfect invariance:

```math
F_b(X)
=
z_0
\quad
\forall X,
```

but it also has maximal failure to respond to every task-changing
intervention. The audit must measure both sides.

## 2. Normalize Sensitivity Against Unrelated Pairs

Raw distances are not comparable across feature dimension, normalization, or
model family. Let `X'` be an unrelated sample from a matched tissue stratum.
Define the normalized intervention ratio:

```math
\rho_b(a)
=
\frac{
\mathbb E_X
\left[
    d_b
    \left(
        F_b(X),
        F_b(aX)
    \right)
\right]
}{
\mathbb E_{X,X'}
\left[
    d_b
    \left(
        F_b(X),
        F_b(X')
    \right)
\right]
}.
```

For an allowed nuisance transform, a strong invariant contract seeks:

```math
\rho_b(a_0)
\ll
1.
```

For a diagnostic intervention, useful sensitivity seeks a ratio separated from
the nuisance distribution:

```math
\Pr
\left[
    d_b
    \left(
        F_b(X),
        F_b(a_1X)
    \right)
    >
    d_b
    \left(
        F_b(X),
        F_b(a_0X)
    \right)
\right].
```

This probability is an intervention-discrimination statistic. It avoids
declaring a large raw distance good when the entire representation has an
arbitrarily large scale.

## 3. Audit The Exported Object, Not A Convenient Surrogate

Models in this family export different objects:

```math
\begin{aligned}
\text{patch vector:}
&\quad
h=f_\phi(x),
\\
\text{patch set:}
&\quad
H_i=\{h_{ij}\}_{j=1}^{n_i},
\\
\text{slide vector:}
&\quad
z_i=F_\psi(H_i,G_i),
\\
\text{multitoken slide object:}
&\quad
Z_i=[z_{i1},\ldots,z_{im}],
\\
\text{retrieval system:}
&\quad
\mathcal N_K(i)
=
\mathcal N_K
\left(
    z_i,
    \mathcal M,
    d
\right).
\end{aligned}
```

An audit of one object does not automatically validate another. For example:

```math
\mathrm{Pool}_{1}(H)
\neq
\mathrm{Pool}_{128}(H),
```

and:

```math
\frac{1}{n}
\sum_jh_j
\neq
\mathrm{CLS}
\left(
    \mathrm{Transformer}(H)
\right).
```

If the paper evaluates a mean patch vector, an attention-pooled slide, and a
fine-tuned slide token, the failure matrix needs three rows or an explicit
operator index.

## 4. Layerwise Bottleneck Localization

Suppose the representation is a composition:

```math
F
=
F_L
\circ
F_{L-1}
\circ
\cdots
\circ
F_1.
```

For paired inputs `X` and `X'`, define layerwise changes:

```math
\Delta_\ell
=
d_\ell
\left(
    F_\ell\circ\cdots\circ F_1(X),
    F_\ell\circ\cdots\circ F_1(X')
\right).
```

The attenuation ratio at layer `ell` is:

```math
A_\ell
=
\frac{
    \Delta_\ell
}{
    \Delta_{\ell-1}+\varepsilon
}.
```

A small ratio identifies the stage where an intervention is compressed. For a
hierarchical model, the product:

```math
\prod_{\ell=1}^{L}A_\ell
```

measures end-to-end attenuation only when the stage metrics are made
comparable. The stagewise curve is generally more informative than one final
distance.

For differentiable models, local influence can also be measured by:

```math
J_{r\to z}
=
\frac{
    \partial z
}{
    \partial h_r
}.
```

Small Jacobian norm is a local statement. A deletion intervention measures a
finite change and can expose nonlinear effects that one Jacobian misses.

## 5. Neighborhood And Memory Audits

For a representation archive:

```math
\mathcal M
=
\{z_1,\ldots,z_M\},
```

define the `K`-neighbor set:

```math
\mathcal N_K(z;\mathcal M)
=
\underset{
    A\subseteq[M],
    |A|=K
}{\arg\min}
\sum_{r\in A}
d(z,z_r).
```

Neighbor stability under intervention `a` is:

```math
\mathrm{Jaccard}_K(a)
=
\frac{
\left|
    \mathcal N_K(F(X);\mathcal M)
    \cap
    \mathcal N_K(F(aX);\mathcal M)
\right|
}{
\left|
    \mathcal N_K(F(X);\mathcal M)
    \cup
    \mathcal N_K(F(aX);\mathcal M)
\right|
}.
```

Archive sensitivity is measured by replacing `M` with a perturbed archive:

```math
\Delta_{\mathrm{archive}}
=
1
-
\frac{
\left|
    \mathcal N_K(z;\mathcal M)
    \cap
    \mathcal N_K(z;\mathcal M')
\right|
}{K}.
```

Useful archive perturbations include:

```text
remove same-patient slides;
remove same-institution slides;
deduplicate near-identical tissue;
balance common and rare diagnoses;
replace one acquisition domain;
remove one semantic group or centroid.
```

For labeled audits, neighborhood purity is:

```math
\mathrm{Purity}_K(z)
=
\frac{1}{K}
\sum_{r\in\mathcal N_K(z)}
\mathbf 1
\{y_r=y_z\}.
```

Purity depends on archive prevalence. A balanced class-conditional audit and a
natural-prevalence audit answer different questions.

## 6. Readout-Specific Interventions

### 6.1 Mean Readout

```math
z^{\mathrm{mean}}
=
\frac{1}{n}
\sum_{j=1}^{n}h_j.
```

Every perturbation preserving the first moment is invisible:

```math
\sum_jh_j
=
\sum_jh_j'
\Longrightarrow
z^{\mathrm{mean}}(H)
=
z^{\mathrm{mean}}(H').
```

### 6.2 Top-K Readout

For class score `s_j(c)`:

```math
S_c^{(K)}
=
\frac{1}{K}
\sum_{j\in\mathrm{TopK}_c}
s_j(c).
```

Its audit varies both `K` and low-score prevalence. If the top `K` scores remain
fixed, all changes outside the selected set are invisible.

### 6.3 Vote-Ratio Readout

```math
r_c
=
\frac{1}{n}
\sum_{j=1}^{n}
\mathbf 1
\{\widehat y_j=c\}.
```

This readout preserves hard class frequency, discards confidence magnitude,
and is invariant to every coordinate permutation.

### 6.4 Attention Readout

```math
z^{\mathrm{attn}}
=
\sum_{j=1}^{n}
\alpha_jh_j,
\qquad
\alpha_j\ge0,
\qquad
\sum_j\alpha_j=1.
```

Deletion should recompute the denominator. Holding the old attention weights
after removal audits a different frozen-weight operator.

### 6.5 Class-Token Readout

```math
z^{\mathrm{CLS}}
=
H_{mathrm{CLS}}^{(L)}.
```

The audit must perturb the token sequence and rerun all context layers. Reading
only the original attention map does not measure how the class token changes.

## 7. Cross-Modal Audits

For normalized image and text vectors:

```math
s(x,q)
=
u(x)^{\mathsf T}v(q).
```

### 7.1 Prompt Dispersion

For class `c` with `R` prompts:

```math
\overline v_c
=
\frac{
    \sum_{r=1}^{R}v(q_{cr})
}{
    \left\|
        \sum_{r=1}^{R}v(q_{cr})
    \right\|_2
},
```

```math
\mathrm{Disp}(c)
=
\frac{1}{R}
\sum_{r=1}^{R}
\left\|
    v(q_{cr})-\overline v_c
\right\|_2^2.
```

The prediction turnover rate is:

```math
\mathrm{Turnover}
=
\Pr
\left[
    \arg\max_{c}
    s(x,q_{cr})
    \ne
    \arg\max_{c}
    s(x,q_{cr'})
\right].
```

### 7.2 Negation Test

For a concept phrase and its negation:

```math
m_{\mathrm{neg}}(x,c)
=
s(x,q_c^{+})
-
s(x,q_c^{-}).
```

The expected sign should reverse between concept-positive and concept-negative
images. A model that maps both prompts to nearly the same direction has lexical
alignment without reliable compositional negation.

### 7.3 Report-Style Intervention

Let `r` be a report and let `a_style(r)` preserve findings while changing
institutional template, section order, or boilerplate. A morphology-aligned
slide-report vector should satisfy:

```math
s(z_{\mathrm{slide}},v(r))
\approx
s
\left(
    z_{\mathrm{slide}},
    v(a_{\mathrm{style}}(r))
\right).
```

This intervention must preserve clinical content. Uncontrolled paraphrasing can
change meaning and invalidate the audit.

## 8. CTransPath Audit Contract

CTransPath uses a momentum contrastive system with semantically relevant
pseudo-positives mined from a memory. Let:

```math
\mathcal P_S(z)
=
\left{
    c_{(1)},\ldots,c_{(S)}
\right}
```

be the `S` most similar memory vectors used as additional positive support.

### 8.1 Preserved Statistic

```text
augmentation-stable local morphology;
shifted-window patch context;
cosine neighborhood relation to the mined memory support.
```

### 8.2 Direct Audit

With audit labels `ell`, mined-support purity is:

```math
\mathrm{Purity}_S(z)
=
\frac{1}{S}
\sum_{q\in\mathcal P_S(z)}
\mathbf 1
\{\ell(q)=\ell(z)\}.
```

The label must match the claimed factor. Tissue-type purity does not validate a
molecular or prognostic neighborhood.

Memory staleness can be measured by storing the encoder step `r_q` for each key:

```math
c_q
=
g_{\xi^{(r_q)}}(x_q),
\qquad
r_q<r.
```

Compare the old and current key:

```math
\Delta_{\mathrm{stale}}(q)
=
1
-
\frac{
    c_q^{\mathsf T}
    g_{\xi^{(r)}}(x_q)
}{
    \|c_q\|_2
    \left\|g_{\xi^{(r)}}(x_q)\right\|_2
}.
```

### 8.3 Falsifying Interventions

```text
stain-preserving augmentation:
    representation and mined support should remain stable;

small-lesion removal:
    representation should change when the lesion is the target factor;

same-site archive removal:
    diagnostic neighborhood should remain useful if site is not the shortcut;

same-slide negative audit:
    repeated morphology should not be treated as semantically unrelated merely
    because patch indices differ;

coordinate permutation:
    the patch encoder should be unchanged because WSI layout is not in its input.
```

### 8.4 Failure Signature

Low view distance together with low lesion sensitivity indicates augmentation
erasure. High mined-support purity for site but low purity for diagnosis
indicates shortcut geometry. High neighbor turnover after recomputing current
keys indicates a stale-memory training signal.

## 9. RetCCL Audit Contract

RetCCL combines weighted instance contrast, cross-branch centroid targets,
feature-spatial mosaic selection, and diagnosis-aware retrieval.

Let branch `p` assign a projected feature to a subqueue:

```math
\kappa_p(z)
=
\arg\max_{r\in\{1,\ldots,S\}}
z^{\mathsf T}\mu_r^{(p)}.
```

The centroid margin is:

```math
m_{\mathrm{cent}}(z)
=
z^{\mathsf T}\mu_{(1)}
-
z^{\mathsf T}\mu_{(2)},
```

where the indices denote the highest and second-highest similarities.

### 9.1 Preserved Statistic

```text
angular patch geometry shaped by weighted negatives;
cross-branch centroid relation;
selected feature-spatial mosaic representatives;
archive-relative diagnosis distribution for query bags.
```

### 9.2 Cluster Stability

For paired views:

```math
\mathrm{Stab}_{\kappa}
=
\Pr
\left[
    \kappa_p(F(T_1x))
    =
    \kappa_p(F(T_2x))
\right].
```

Condition this statistic on centroid margin. High stability caused only by a
few dominant tissue or stain clusters does not establish fine morphology.

### 9.3 Mosaic Coverage

Let `D_i` be a set of independently annotated diagnostic patches and `M_i` the
selected mosaic. Coverage within feature radius `epsilon` is:

```math
\mathrm{Coverage}_{\varepsilon}(i)
=
\frac{1}{|D_i|}
\sum_{d\in D_i}
\mathbf 1
\left{
    \min_{m\in M_i}
    \|F(d)-F(m)\|_2
    \le
    \varepsilon
\right}.
```

Low coverage identifies selection loss before retrieval begins.

### 9.4 Archive Audit

RetCCL's final query-bag distribution uses archive diagnosis metadata. Rebuild
the archive under patient exclusion and class balancing, then measure:

```math
\left\|
    p_i(\cdot;\mathcal M)
    -
    p_i(\cdot;\mathcal M')
\right\|_1.
```

Large change is an archive failure, not necessarily a change in the CCL patch
encoder.

### 9.5 Failure Signature

Small centroid margins with high assignment turnover expose self-label
instability. High patch-level retrieval but low mosaic coverage exposes
selection bias. Stable diagnosis only before patient deduplication exposes
memory leakage.

## 10. HIPT Audit Contract

Write the hierarchical map as:

```math
z_i
=
F_3
\left(
    \left{
        F_2
        \left(
            \{F_1(x_{ikj})\}_{j=1}^{256}
        \right)
    \right}_{k=1}^{M_i}
\right).
```

Each stage exports a class-token bottleneck.

### 10.1 Preserved Statistic

```text
DINO-stable local patch morphology;
within-region arrangement and interaction at the 4096 scale;
task-adapted composition of region tokens at slide scale.
```

### 10.2 Boundary-Shift Audit

Let `B_delta` translate the 4096-region grid by offset `delta` while keeping the
underlying slide pixels fixed. Compare:

```math
\Delta_{\mathrm{boundary}}(\delta)
=
d
\left(
    F_{\mathrm{HIPT}}(X;B_0),
    F_{\mathrm{HIPT}}(X;B_\delta)
\right).
```

The audit is especially important for lesions or tumor-stroma interfaces that
cross a region boundary under one partition and remain inside a region under
another.

### 10.3 Bottleneck Attenuation

Inject or remove a local target feature and record:

```math
\Delta_1,
\qquad
\Delta_2,
\qquad
\Delta_3.
```

If:

```math
\Delta_1>0,
\qquad
\Delta_2\approx0,
```

the region class token removed the signal. If:

```math
\Delta_2>0,
\qquad
\Delta_3\approx0,
```

the slide module or final readout removed it.

### 10.4 Position Audit

Permute region tokens while preserving their values:

```math
R_i'
=
\Pi R_i.
```

Under no explicit WSI coordinate side channel and a permutation-equivariant
context implementation:

```math
F_3(\Pi R_i)
=
F_3(R_i)
```

for class-token readout. This tests the reported global position contract; it
does not test within-region positional encoding at stage 2.

### 10.5 Failure Signature

Strong boundary dependence identifies hierarchy partition bias. Local changes
that disappear at one class-token stage identify the exact compression level.
Stable slide output under region-coordinate permutation confirms that absolute
global layout is absent rather than proving spatial robustness.

## 11. UNI And Virchow Audit Contract

UNI and Virchow export patch vectors, not complete WSI representations:

```math
h_{ij}^{(b)}
=
f_{\phi_b}(x_{ij}),
\qquad
b\in\{\mathrm{UNI},\mathrm{Virchow}\}.
```

The original UNI feature is a class-token statistic. Virchow exports:

```math
e^{\mathrm{Virchow}}(x)
=
\left[
    c(x)
    ;
    \overline p(x)
\right],
\qquad
\overline p(x)
=
\frac{1}{256}
\sum_{r=1}^{256}p_r(x).
```

### 11.1 Preserved Statistic

```text
augmentation-stable patch morphology;
class-token crop summary;
for Virchow, an additional first moment of local crop tokens;
no WSI coordinate relation in one exported patch vector.
```

### 11.2 Augmentation Orbit Audit

For the actual view kernel `K_b`:

```math
\mathcal E_{b}^{\mathrm{view}}
=
\mathbb E_{x,T_1,T_2\sim K_b(\cdot\mid x)}
\left[
    \left\|
        f_b(T_1x)
        -
        f_b(T_2x)
    \right\|_2^2
\right].
```

Pair this with a target-feature deletion audit. Small view error and small
deletion response indicate that the feature lies inside the learned nuisance
quotient.

### 11.3 Virchow Block Audit

For downstream linear map:

```math
W[c;\overline p]
=
W_{\mathrm{CLS}}c
+
W_{\mathrm{patch}}\overline p.
```

Report block influence:

```math
D(x)
=
\|W_{\mathrm{CLS}}c\|_2
+
\|W_{\mathrm{patch}}\overline p\|_2,
\qquad
D(x)>0,
```

```math
r_c(x)
=
\frac{\|W_{\mathrm{CLS}}c\|_2}{D(x)},
\qquad
r_p(x)
=
\frac{\|W_{\mathrm{patch}}\overline p\|_2}{D(x)},
\qquad
r_c(x)+r_p(x)=1.
```

For retrieval, separately normalize the two blocks and compare ranking
turnover. Concatenation does not give both blocks equal metric weight.

### 11.4 WSI Layout Negative Control

Permute slide coordinates while preserving every patch image:

```math
\{(x_j,c_j)\}
\longrightarrow
\{(x_j,c_{\pi(j)})\}.
```

The patch vectors should remain identical. A downstream set readout should also
remain identical. Any claimed layout sensitivity must enter through a separate
coordinate-aware slide operator.

### 11.5 Failure Signature

Strong patch probes with layout-blind WSI predictions identify a downstream
geometry gap, not a patch-encoder defect. Large Virchow ranking changes after
block normalization identify metric weighting. Rare features absent from the
sampled patch support are coverage failures no downstream readout can repair.

## 12. PLIP And CONCH Audit Contract

PLIP learns a CLIP-style normalized image-text geometry. CONCH additionally
uses a separate 256-query image pooler for autoregressive captioning and uses
class-specific top-`K` WSI scores in its MI-Zero evaluation.

### 12.1 Preserved Statistic

```text
PLIP:
    OpenPath image-caption cosine geometry;

CONCH contrastive path:
    one-query image-text retrieval vector;

CONCH caption path:
    256-query image token set conditioned into a text decoder;

CONCH WSI zero-shot path:
    class-specific upper order statistics over independent tile scores.
```

### 12.2 Prompt And Negation Audit

Measure prompt dispersion, decision turnover, and negation margin from Section
7. Class-name synonyms should be stable when they preserve meaning. Negation,
grade changes, and explicit absence should change the semantic direction.

### 12.3 Pooler Identity Audit

Do not compare:

```math
u_i
=
\mathrm{Pool}_{1}(H_i)
```

with:

```math
C_i
=
\mathrm{Pool}_{256}(H_i)
```

as though they were two views of one vector. Compare task behavior after the
correct downstream reader for each exported object.

### 12.4 Top-K Audit

Evaluate the curve:

```math
K
\longmapsto
S_{i,c}^{(K)}.
```

Construct paired slides with equal top-`K` scores and different positive-tile
prevalence. The top-`K` classifier must collide on that pair. This is a readout
property, not a failure of the image-text encoder.

### 12.5 Spatial Negative Control

Permute tile coordinates while holding tile scores fixed. MI-Zero top-`K`
prediction should be exactly unchanged:

```math
S_{i,c}^{(K)}
\left(
    \{s_{ijc},c_{ij}\}
\right)
=
S_{i,c}^{(K)}
\left(
    \{s_{ijc},c_{i\pi(j)}\}
\right).
```

### 12.6 Failure Signature

High prompt turnover identifies text-direction instability. Stable similarity
for a phrase and its negation identifies semantic composition failure. High
zero-shot performance that collapses under a new `K` identifies a sparse-
evidence readout assumption. Spatial invariance is expected for MI-Zero and
must not be misreported as slide-level spatial understanding.

## 13. Prov-GigaPath Audit Contract

Prov-GigaPath composes a frozen DINOv2 tile encoder with a LongNet slide
encoder, coordinate serialization, masked slide-token reconstruction, and a
downstream shallow ABMIL reader.

### 13.1 Preserved Statistic

```text
local DINOv2 tile morphology;
coordinate-indexed contextual tile states;
long-range interactions available through dilated attention paths;
tile content predictable from unmasked slide context;
task-specific weighted first moment after downstream ABMIL.
```

### 13.2 Coordinate-Contract Audit

For coordinate transform `g`:

```math
\Delta_g
=
d
\left(
    F
    \left(
        \{h_j,c_j\}
    \right),
    F
    \left(
        \{h_j,g(c_j)\}
    \right)
\right).
```

The paper's slide augmentations include crop, translation within the coordinate
grid, and horizontal flips. Test those separately from unsupported rotations,
arbitrary scale changes, and retessellation.

### 13.3 Quantization Collision

With grid map:

```math
q(c)
=
\left\lfloor
    \frac{c}{256}
\right\rfloor,
```

two coordinates collide when:

```math
q(c)
=
q(c')
\quad
\text{while}
\quad
c\ne c'.
```

Construct sub-grid translations and verify whether the representation is
exactly unchanged or changes only through tile content.

### 13.4 Sparse-Support Audit

Let `S_l(i)` be the dilated attention support of token `i` at layer `l`. Define
the earliest path length between tiles:

```math
L^{\star}(i,j)
=
\min
\left{
    L:
    j
    \text{ is reachable from }
    i
    \text{ through }L\text{ support steps}
\right}.
```

Intervention influence should be stratified by path length:

```math
I_L
=
\mathbb E
\left[
    \left\|
        z(X)-z(X\setminus j)
    \right\|_2
    \mid
    L^{\star}(\mathrm{CLS},j)=L
\right].
```

### 13.5 Mask Ambiguity And Tile Deletion

For masked token `j`, the slide pretraining target is a tile embedding. Measure
reconstruction uncertainty across contexts with similar visible tokens. For
downstream behavior, delete rare regions and common regions at matched area:

```math
\Delta_{\mathrm{rare}}
=
d(z(X),z(X\setminus R_{\mathrm{rare}})),
```

```math
\Delta_{\mathrm{common}}
=
d(z(X),z(X\setminus R_{\mathrm{common}})).
```

### 13.6 Readout And Fine-Tuning Audit

The source paper freezes the tile encoder, fine-tunes the LongNet slide
encoder, and trains shallow ABMIL plus a classifier for downstream tasks.
Compare:

```text
frozen tile plus frozen slide plus linear head;
frozen tile plus fine-tuned slide plus ABMIL;
frozen tile plus randomly initialized slide plus ABMIL;
class-token readout versus contextual-tile ABMIL.
```

This separates fixed representation, pretrained initialization, slide
architecture, and downstream readout.

### 13.7 Failure Signature

Large change under training-declared coordinate nuisances indicates invariance
failure. Zero change under clinically meaningful rearrangement indicates
geometry blindness. Influence decaying sharply with sparse path length exposes
long-range support limits. Rare-region deletion with negligible output change
exposes a mask, context, or readout bottleneck.

## 14. TITAN Audit Contract

TITAN uses frozen CONCHv1.5 patch features, vision-only feature-grid
pretraining, ROI-caption alignment, WSI-report alignment, one-query retrieval
pooling, and 128-query generation pooling.

### 14.1 Preserved Statistic

```text
CONCHv1.5 local morphology;
within-grid spatial context from vision pretraining;
fine-grained ROI-caption alignment;
coarse WSI-report alignment;
one-query global retrieval and zero-shot vector;
128-query conditional generation state.
```

### 14.2 Stage-Ablation Audit

Let the three objective stages be indexed by:

```math
\mathcal L_1,
\qquad
\mathcal L_2,
\qquad
\mathcal L_3.
```

For task metric `M`, define marginal stage effects relative to matched training:

```math
\Delta_r^{\mathrm{stage}}
=
\mathcal M(F_{1,2,3})
-
\mathcal M(F_{\{1,2,3\}\setminus r}).
```

The source paper reports stage ablations for zero-shot performance. The audit
should also examine morphology-only probes and external retrieval because a
language stage can improve text alignment while changing visual neighborhood
structure.

### 14.3 Train-Short, Test-Long Audit

Stage 1 sees bounded feature-grid crops, while inference can use larger WSI
grids. For nested crop sizes `m`:

```math
z_i^{(m)}
=
F
\left(
    H_i^{(m)}
\right).
```

Measure consistency on the overlapping tissue:

```math
\Delta_{m,m'}
=
d
\left(
    z_i^{(m)},
    z_i^{(m')}
\right).
```

The expected change is not zero because added tissue is real information. The
control compares added diagnostic tissue with matched added background or
nondiagnostic tissue.

### 14.4 One-Query Versus 128-Query Audit

```math
z_i^{(1)}
=
\mathrm{Pool}_{1}(H_i),
\qquad
Z_i^{(128)}
=
\mathrm{Pool}_{128}(H_i).
```

Use the one-query vector for retrieval and zero-shot audits. Use the 128-token
object for generation audits. A good generated report does not prove that the
one-query vector preserves every reported finding.

### 14.5 Report And Synthetic-Caption Audit

Apply content-preserving report-style transformations, remove boilerplate, and
stratify by whether a phrase was generated by PathChat or written in a clinical
report. For matched visual content, measure:

```math
\Delta_{\mathrm{text\mbox{-}source}}
=
s(z_i,t_i^{\mathrm{clinical}})
-
s(z_i,t_i^{\mathrm{synthetic}}).
```

This does not declare either source correct. It exposes which text distribution
defines the alignment.

### 14.6 Archive And External Audit

Recompute slide retrieval after patient exclusion, institution exclusion, and
class balancing. TITAN's external rare-cancer retrieval cohort provides a
direct domain-shift test; report case count, diagnosis composition, and
uncertainty rather than only one aggregate retrieval score.

### 14.7 Survival Evaluation Audit

The source paper evaluates a linear Cox head on one slide embedding. Its methods
state that regularization is selected by the best average test metric across
folds. If the same fold outcomes select and report the hyperparameter, the
selected estimate obeys the winner's-curse tendency:

```math
\mathbb E
\left[
    \max_r\widehat M_r
\right]
\ge
\max_r
\mathbb E
\left[
    \widehat M_r
\right].
```

Nested tuning is the direct audit. This is a survival-evaluation limitation,
not evidence against the slide representation.

### 14.8 Failure Signature

Language gains without morphology-probe gains identify alignment-specific
transfer. Large crop-context instability after matched nondiagnostic expansion
identifies extrapolation sensitivity. Disagreement between one-query retrieval
and 128-token generation identifies a readout bottleneck. Retrieval collapse
after archive or institution exclusion identifies memory-relative geometry.

## 15. KEEP Audit Contract

KEEP uses a disease graph in knowledge-encoder pretraining, semantic-group
construction, and hypernym-aware negative filtering. Its zero-shot WSI reader
is a tile classifier followed by a tumor-ratio statistic.

### 15.1 Preserved Statistic

```text
disease-attribute metric ordering;
semantic-group image-text relation;
hypernym-aware exclusion of related negatives;
tile-level prompt geometry;
hard tile-class frequency at WSI readout.
```

### 15.2 Knowledge-Graph Edge Audit

Let the disease graph be:

```math
\mathcal G
=
(\mathcal D,\mathcal E,\mathcal A).
```

For edge `e`, retrain or reevaluate the text-side relation after deletion:

```math
\Delta_e
=
d
\left(
    \Phi_k^{\mathcal G}(a),
    \Phi_k^{\mathcal G\setminus e}(a)
\right).
```

When full retraining is infeasible, at minimum recompute the negative mask and
measure which group pairs change status. A posthoc embedding perturbation is not
equivalent to retraining the graph-conditioned objective.

### 15.3 Hypernym-Negative Audit

Let `I_ij` indicate whether group `j` is allowed as a negative for group `i`.
Compare:

```math
I_{ij}^{\mathrm{KEEP}}
```

with unrestricted negatives:

```math
I_{ij}^{\mathrm{all}}
=
\mathbf 1\{i\ne j\}.
```

Measure false-negative rate using independent expert relations and report the
effect on rare and neighboring diseases separately.

### 15.4 Held-Out Concept Audit

Partition disease entities so that a test entity and its direct attributes do
not appear in training. Evaluate whether hierarchy relations support transfer
without leaking synonyms or near-duplicate captions.

The relevant split is over semantic entities and patients, not random
image-caption pairs.

### 15.5 Tumor-Ratio Audit

KEEP's WSI statistic is:

```math
\widehat r_i(c)
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
    \mathbf 1
    \left[
    \arg\max_{a}
    q_{ij}(a)
    =c
    \right].
```

Audit soft versus hard prevalence:

```math
\widetilde r_i(c)
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
q_{ij}(c).
```

Their difference is:

```math
\Delta_{\mathrm{hard}}
=
\left|
    \widehat r_i(c)
    -
    \widetilde r_i(c)
\right|.
```

Large difference identifies confidence discarded by hard voting. Coordinate
permutation must leave both ratios unchanged.

### 15.6 Failure Signature

Large performance change after one plausible graph-edge correction identifies
prior fragility. Gains that disappear on entity-held-out evaluation identify
semantic leakage. Stable tile accuracy with unstable tumor ratio identifies a
WSI readout problem. Strong ratio prediction with coordinate-permutation
invariance confirms prevalence modeling, not spatial reasoning.

## 16. Distillation Audit Contract

For teacher and student distributions at temperature `tau`:

```math
p_t^{(\tau)}(x),
\qquad
p_s^{(\tau)}(x),
```

the output distillation loss can be:

```math
\mathcal L_{\mathrm{KD}}
=
\tau^2
D_{\mathrm{KL}}
\left(
    p_t^{(\tau)}
    \,\|\,
    p_s^{(\tau)}
\right).
```

Feature and relation matching add different contracts:

```math
\mathcal L_{\mathrm{feat}}
=
\left\|
    A_sh_s-A_th_t
\right\|_2^2,
```

```math
\mathcal L_{\mathrm{rel}}
=
\left\|
    K_s(H_s)-K_t(H_t)
\right\|_F^2.
```

### 16.1 Preserved Statistic

```text
teacher class distribution under output matching;
projected teacher feature under feature matching;
teacher pair geometry under relation matching;
teacher multimodal relation under modality matching.
```

### 16.2 Teacher-Blind-Spot Audit

Partition the test set into teacher-correct and teacher-incorrect strata:

```math
\mathcal D_{+}
=
\{i:\widehat y_t(x_i)=y_i\},
\qquad
\mathcal D_{-}
=
\{i:\widehat y_t(x_i)\ne y_i\}.
```

Report student agreement and task accuracy in both. High teacher-student
agreement on the teacher-error stratum is faithful compression of a teacher
error.

### 16.3 Relation Audit

Output matching can succeed while neighborhood geometry changes. Compare
teacher and student Gram matrices:

```math
G_t
=
H_tH_t^{\mathsf T},
\qquad
G_s
=
H_sH_s^{\mathsf T}.
```

The normalized relation discrepancy is:

```math
\Delta_G
=
\frac{
    \|G_s-G_t\|_F
}{
    \|G_t\|_F+\varepsilon
}.
```

Audit retrieval, calibration, rare morphology, and slide context separately.

### 16.4 Failure Signature

Matched average logits with poor subgroup calibration indicates compressed
uncertainty. Matched logits with changed neighbors identifies relation loss.
Matched teacher behavior on teacher-error strata identifies inherited blind
spots rather than successful pathology correction.

## 17. Cross-Paper Failure Matrix

| Model family | Exported or evaluated statistic | Direct intervention | Expected behavior | Failure signature |
|---|---|---|---|---|
| CTransPath | patch vector and mined cosine support | lesion-preserving view versus lesion deletion; archive replacement | nuisance stability, target sensitivity, diagnostic support purity | false pseudo-positive, stale memory, augmentation erasure, site shortcut |
| RetCCL | cluster-shaped patch geometry, mosaic, diagnosis-aware query bag | centroid perturbation, mosaic coverage, archive rebalance | stable high-margin assignments and diagnostic coverage | self-label loop, rare-patch omission, archive dependence |
| HIPT | nested patch, region, and slide class tokens | hierarchy boundary shift and local lesion insertion | stable nuisance partition, measurable stagewise lesion survival | region-boundary bias or class-token attenuation |
| UNI | crop class token and downstream patch set | view orbit, target deletion, coordinate permutation | patch invariance with target sensitivity; layout negative control | local feature erasure or downstream geometry gap |
| Virchow | crop class token plus local-token mean | block normalization and patch audit | stable target geometry across justified normalization | one block dominates retrieval or readout |
| PLIP | normalized image-text vector | prompt paraphrase, negation, pair-domain shift | semantic stability and polarity sensitivity | prompt shortcut or pair-distribution shift |
| CONCH | one-query retrieval vector, 256-query caption tokens, top-K WSI score | pooler identity, prompt, `K`, spatial permutation | task-specific token behavior and exact top-K invariance | pooler confusion, prevalence blindness, prompt instability |
| Prov-GigaPath | contextual slide tokens and downstream ABMIL statistic | coordinate transform, retile, sparse-path deletion, reader swap | training-declared coordinate stability and long-range influence | quantization, serialization, sparse support, mask or reader bottleneck |
| TITAN | one-query slide vector, 128 generation tokens, retrieval archive | stage ablation, crop scale, report style, archive/site exclusion | task-specific stage gains and external semantic stability | crop extrapolation, report shortcut, readout mismatch, archive dependence |
| KEEP | knowledge-conditioned image-text geometry and tumor ratio | graph edge, hypernym mask, entity holdout, hard versus soft ratio | stable valid hierarchy transfer and prevalence behavior | knowledge prior error, semantic leakage, hard-vote loss |
| distillation | teacher output, feature, relation, or multimodal target | teacher-error stratum and relation comparison | explicit preservation of the chosen teacher object only | inherited blind spots, relation loss, uncertainty compression |

The matrix deliberately uses "expected behavior" rather than "should always be
invariant." Some interventions should change the representation. A system that
ignores every perturbation is collapsed, not robust.

## 18. Statistical Design For Intervention Audits

### 18.1 Pair At The Patient Level

For patient `i`, define paired intervention effect:

```math
D_i
=
m
\left(
    F(X_i),
    F(aX_i)
\right).
```

The estimator is:

```math
\widehat\Delta
=
\frac{1}{N}
\sum_{i=1}^{N}D_i.
```

Its standard error is estimated from patient-level variation:

```math
\widehat{\mathrm{SE}}
\left(
    \widehat\Delta
\right)
=
\sqrt{
    \frac{
        \widehat{\mathrm{Var}}(D_i)
    }{N}
}.
```

Treating thousands of patches from one patient as independent would replace
biological replication with technical replication.

### 18.2 Stratify The Intervention

Report effects by:

```text
tissue and disease;
rare versus common morphology;
institution and scanner;
patch, region, slide, and patient level;
pretraining-overlap status;
target prevalence and slide area.
```

An average near zero can hide equal and opposite subgroup effects.

### 18.3 Multiple Audit Selection

If many interventions are searched and only the largest effect is reported:

```math
\widehat a
=
\arg\max_{a\in\mathcal A}
|\widehat\Delta_a|,
```

then:

```math
\mathbb E
\left[
    \max_{a\in\mathcal A}
    |\widehat\Delta_a|
\right]
```

is upward biased relative to any prespecified single effect. Prespecify core
interventions or correct for multiplicity.

### 18.4 Equivalence Requires A Margin

Failure to reject a difference is not evidence of invariance. Define a
practical equivalence margin `epsilon` and test whether the confidence interval
lies inside:

```math
[-\varepsilon,+\varepsilon].
```

The margin must be stated in a meaningful metric, such as probability change,
rank turnover, embedding distance relative to unrelated pairs, or task-risk
change.

## 19. Confounds That Can Fake A Successful Audit

### 19.1 Intervention Changes Image Quality

Lesion removal, stain normalization, and retessellation can introduce visible
seams or interpolation artifacts. A model may respond to the editing artifact
rather than the intended tissue factor. Use sham edits and matched controls.

### 19.2 The Label Changes With The Intervention

A content-preserving nuisance transform must genuinely preserve the target. A
crop that removes the only tumor focus is not a nuisance intervention.

### 19.3 Archive Leakage

Same-patient, same-block, or near-duplicate slides can make retrieval appear
semantically stable. Deduplicate before testing archive geometry.

### 19.4 Readout Is Not Recomputed

After deleting a patch, attention, class tokens, graph edges, and normalization
must be recomputed. Reusing old weights tests a first-order surrogate.

### 19.5 Pretraining Overlap

An external-looking benchmark can overlap the pretraining institution, patient,
or public corpus. Separate known overlap, possible overlap, and verified
non-overlap.

### 19.6 Metric Choice Hides The Failure

AUROC can remain high while calibration changes. Recall at `K` can remain high
while neighbor identity changes. Average accuracy can remain high while a rare
subgroup collapses. Pair every audit metric with the failure it can actually
detect.

## 20. C/R/G/S Audit Ledger

Each audit should record both pretraining and downstream operators:

| Field | Required audit entry |
|---|---|
| `C_pre` | patch window, teacher-student views, memory mining, hierarchy, slide attention, cross-modal decoder, or knowledge encoder |
| `R_pre` | projection vector, class token, query token set, masked prediction target, or teacher relation |
| `G_pre` | augmentation quotient, cosine metric, cluster geometry, coordinates, hierarchy, ALiBi, text space, or disease graph |
| `S_pre` | contrastive pairs, generated centroids, teacher targets, masked tokens, captions, reports, or knowledge attributes |
| `C_task` | identity, MIL context, LongNet fine-tuning, patient aggregation, prompt conditioning, or memory lookup |
| `R_task` | mean, attention, top-K, vote ratio, class token, prototype, or retrieval set |
| `H_task` | linear softmax, Cox, similarity decision, classifier, or decoder |
| surviving statistic | exact vector, moment, order statistic, token set, neighborhood, or sequence measured by the audit |

The intervention then names which field it changes. Examples:

```text
prompt paraphrase:
    changes G_task while holding image representation fixed;

coordinate permutation:
    changes G_task or G_pre input geometry;

archive replacement:
    changes R_task support while holding the encoder fixed;

stage ablation:
    changes C_pre and S_pre;

fine-tuning ablation:
    changes the arrow from S_task into C_pre parameters.
```

This prevents attributing every failure to the foundation encoder.

## 21. Minimal Paper-Specific Failure Report

For each claimed failure or robustness result, report:

```text
1. source paper and checkpoint;
2. exact exported object and tensor shape;
3. frozen and trainable components;
4. pretraining relation being tested;
5. nuisance transformation family;
6. task-changing intervention family;
7. context, readout, geometry, supervision, and task head;
8. patient-level sample size and strata;
9. matched sham intervention;
10. distance, task metric, or neighborhood statistic;
11. practical equivalence margin;
12. confidence interval and multiplicity rule;
13. archive, prompt, coordinate, and normalization details;
14. result expected under the source-paper contract;
15. alternative explanation for the observed result.
```

A failure claim without an alternative explanation is not yet a diagnostic
audit. It is an observed discrepancy.

## 22. Bottom Line

The paper-specific audit is:

```math
\boxed{
\left(
    \text{source contract},
    \text{exported statistic},
    \text{intervention},
    \text{expected invariant or response},
    \text{failure signature},
    \text{confound}
\right).
}
```

CTransPath and RetCCL require memory and cluster audits. HIPT requires
stagewise hierarchy audits. UNI and Virchow require patch-geometry audits plus
an explicit downstream WSI reader. PLIP and CONCH require prompt, pooler, and
top-`K` audits. Prov-GigaPath requires coordinate, sparse-context, mask, and
fine-tuning audits. TITAN requires stage, scale, report, token-readout, archive,
and survival-protocol audits. KEEP requires graph, semantic-group, and
tumor-ratio audits. Distillation requires teacher-error and relation-preservation
audits.

The decisive question is not:

```math
\text{Did the benchmark score decrease?}
```

It is:

```math
\boxed{
\text{Which mathematical contract failed, at which operator, under which
controlled intervention, and what other mechanism could produce the same
observation?}
}
```
