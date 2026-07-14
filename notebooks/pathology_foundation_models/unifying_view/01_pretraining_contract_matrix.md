# Pretraining Contract Matrix

Primary paper anchors:

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
- [paper-specific derivations](../paper_specific_derivations/README.md)

The phrase "pretrained pathology encoder" is incomplete. The learned map is the
result of a statistical contract specifying what is sampled, what is hidden,
what is paired, what may interact, what is compared, and which object is
exported.

Two models can share an architecture and learn different representations
because their contracts differ. Two models can use different architectures and
still preserve similar task statistics because their contracts induce similar
relations.

## 1. Canonical Contract

Define a pretraining contract as:

```math
\boxed{
\mathfrak P
=
\left(
    \mathcal D,
    \mathcal V,
    \mathcal Q,
    \mathcal C,
    \mathcal R,
    \mathcal H,
    \mathcal L,
    \mathcal G
\right).
}
```

The fields are:

```text
D:
    sampling distribution and independent unit;

V:
    view, crop, mask, corruption, modality, or coordinate kernel;

Q:
    positive, negative, teacher, reconstruction, caption, report, or knowledge
    target law;

C:
    context operator available before the pretraining readout;

R:
    pretraining readout or exported token-selection operator;

H:
    projection, prediction, teacher, contrastive, or decoder head;

L:
    scalar objective and weighting among its terms;

G:
    comparison geometry, such as cosine, Euclidean, tokenwise, hierarchy,
    coordinate, or autoregressive geometry.
```

Let the sampled source object be:

```math
X
\sim
\mathcal D.
```

Views and targets are sampled conditionally:

```math
V
\sim
\mathcal V(\cdot\mid X),
\qquad
Q
\sim
\mathcal Q(\cdot\mid X,V).
```

The pretraining computation is:

```math
\widetilde H_{\theta}
=
\mathcal C_{\theta}
\left(
    X,V;
    \mathcal G
\right),
```

```math
U_{\theta}
=
\mathcal R_{\theta}
\left(
    \widetilde H_{\theta}
\right),
```

```math
P_{\theta}
=
\mathcal H_{\theta}
\left(
    U_{\theta}
\right).
```

The population objective is:

```math
\mathcal J_{\mathfrak P}(\theta)
=
\mathbb E_{
    X\sim\mathcal D,
    V\sim\mathcal V(\cdot\mid X),
    Q\sim\mathcal Q(\cdot\mid X,V)
}
\left[
    \mathcal L
    \left(
        P_{\theta},
        Q;
        \mathcal G
    \right)
\right].
```

Training selects one approximate minimizer:

```math
\widehat\theta
\approx
\arg\min_{\theta}
\widehat{\mathcal J}_{\mathfrak P}(\theta).
```

The exported representation can differ from `P_theta`:

```math
Z
=
\Phi_{\widehat\theta}(X).
```

For example, contrastive training can optimize a projection head while
downstream tasks export the backbone feature. A contract must state both the
optimized object and the exported object.

## 2. Sampling Distribution Is Part Of The Method

For slide `i` with `n_i` patches, a patch-uniform sampler induces:

```math
P_{\mathrm{patch}}(x)
=
\sum_{i=1}^{N}
\frac{n_i}{\sum_{r=1}^{N}n_r}
P_i(x).
```

A slide-uniform sampler induces:

```math
P_{\mathrm{slide\text{-}uniform}}(x)
=
\frac{1}{N}
\sum_{i=1}^{N}
P_i(x).
```

These coincide only under special conditions, such as equal patch counts and
equal within-slide sampling rules. A large slide can dominate patch-uniform
training even though downstream evaluation weights each patient once.

The empirical patch objective is:

```math
\widehat{\mathcal J}_{\mathrm{patch}}
=
\frac{1}{\sum_i n_i}
\sum_{i=1}^{N}
\sum_{j=1}^{n_i}
\ell_{ij}.
```

The slide-uniform objective is:

```math
\widehat{\mathcal J}_{\mathrm{slide}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\ell_{ij}.
```

Even with the same loss and architecture:

```math
\widehat{\mathcal J}_{\mathrm{patch}}
\ne
\widehat{\mathcal J}_{\mathrm{slide}}
```

in general. Data scale stated only as a patch count hides this weighting law.

## 3. The View Kernel Defines Candidate Equivalences

Let two views be sampled from one source object:

```math
V_1,V_2
\sim
\mathcal V(\cdot\mid X).
```

An alignment objective often creates pressure toward:

```math
\Phi(V_1)
\approx
\Phi(V_2).
```

The view kernel first defines a symmetric co-support relation:

```math
v_1
\mathrel{R_{\mathcal V}}
v_2
\quad
\Longleftrightarrow
\quad
\exists x
\text{ such that }
\mathcal V(v_1\mid x)>0
\text{ and }
\mathcal V(v_2\mid x)>0.
```

The co-support relation need not be transitive. Let its equivalence closure be:

```math
\sim_{\mathcal V}
=
\mathrm{EqCl}
\left(
    R_{\mathcal V}
\right).
```

Alignment along sampled pairs creates pressure toward constancy on connected
components of this relation graph. That induced equivalence can be too broad.
If one legal view removes a small lesion and another preserves it, the contract
asks the representation to identify two task-distinct objects.

It can also be too narrow. Two independent patches can contain the same
morphology but enter only as negatives because their indices differ.

Thus the view kernel specifies neither truth nor nuisance by itself. It
specifies which distinctions the objective is rewarded for ignoring.

## 4. The Target Law Defines A Relation Graph

For a batch of source or view nodes:

```math
\mathcal B
=
\{v_1,\ldots,v_B\},
```

define weighted positive and negative relations:

```math
A_{ab}^{+}
\ge
0,
\qquad
A_{ab}^{-}
\ge
0.
```

A general pairwise objective has the form:

```math
\mathcal L_{\mathrm{rel}}
=
\sum_{a,b}
A_{ab}^{+}
\ell_{+}(z_a,z_b)
+
\sum_{a,b}
A_{ab}^{-}
\ell_{-}(z_a,z_b).
```

Different methods construct `A_plus` and `A_minus` differently:

```text
instance contrast:
    two views of one patch are positive;

memory mining:
    high-similarity archive elements become pseudo-positive;

cluster contrast:
    centroid or subqueue assignments change pair weights;

teacher-student learning:
    the teacher output is a directed soft target;

image-text alignment:
    paired image and caption form a cross-modal positive;

knowledge alignment:
    disease groups and hypernym paths modify valid negatives;

distillation:
    teacher outputs, features, or relations define directed targets.
```

The relation graph, not the word "contrastive," determines which points attract
or repel one another.

## 5. Contrastive Contract

For normalized query, positive key, and negative keys:

```math
q
=
\frac{g(f(v_1))}{\|g(f(v_1))\|_2},
\qquad
k^+
=
\frac{g_t(f_t(v_2))}{\|g_t(f_t(v_2))\|_2},
```

the instance InfoNCE loss is:

```math
\mathcal L_{\mathrm{NCE}}
=
-
\log
\frac{
    \exp(q^{\mathsf T}k^+/\tau)
}{
    \exp(q^{\mathsf T}k^+/\tau)
    +
    \sum_{r=1}^{M}
    \exp(q^{\mathsf T}k_r^-/\tau)
}.
```

The direct contract is relative angular separation among sampled supports. It
does not directly constrain:

```text
absolute calibration;
WSI coordinate geometry;
all same-morphology patches to be close;
all task labels to be linearly separable;
all unseen institutions to preserve the same neighborhood.
```

CTransPath expands positive support with memory-mined semantically relevant
patches. RetCCL changes negative weights and adds centroid relations. They are
not identical contracts despite both being contrastive patch pretraining.

## 6. Teacher-Student Contract

For teacher and student view distributions:

```math
q_t(v_g)
=
\mathrm{softmax}
\left(
    \frac{h_t(v_g)-c_t}{\tau_t}
\right),
```

```math
p_s(v)
=
\mathrm{softmax}
\left(
    \frac{h_s(v)}{\tau_s}
\right),
```

a self-distillation term is:

```math
\mathcal L_{\mathrm{distill}}
=
-
\sum_{k=1}^{K}
q_{t,k}(v_g)
\log
p_{s,k}(v).
```

The teacher is updated by an exponential moving average:

```math
\theta_t^{(r)}
=
m_r\theta_t^{(r-1)}
+
(1-m_r)\theta_s^{(r)}.
```

The contract preserves agreement with a moving, centered, sharpened teacher
distribution. It does not reveal a fixed ground-truth class. HIPT uses this
contract at nested spatial levels. UNI and Virchow use DINO-family contracts to
learn broad patch geometry.

## 7. Masked-Prediction Contract

Let token set `M` be masked and `O` be observed. The encoder receives:

```math
X_O
=
\{x_j:j\notin\mathcal M\}.
```

The predictor estimates a target:

```math
\widehat T_{\mathcal M}
=
D_\psi
\left(
    \mathcal C_\phi(X_O,\mathcal M)
\right).
```

For squared reconstruction:

```math
\mathcal L_{\mathrm{mask}}
=
\frac{1}{|\mathcal M|}
\sum_{j\in\mathcal M}
\left\|
    \widehat T_j-T_j
\right\|_2^2.
```

The population optimum under squared loss is the conditional mean:

```math
\widehat T_j^{\star}
=
\mathbb E
\left[
    T_j
    \mid
    X_O,\mathcal M
\right].
```

The directly rewarded statistic is therefore information useful for predicting
the hidden target from visible context. Unpredictable but diagnostic rare
morphology can be averaged away.

Phikon-style masked image modeling applies this logic to patch tokens.
Prov-GigaPath applies masked reconstruction to slide-level tile embeddings, so
its conditional object includes cross-tile slide context.

## 8. Image-Text And Generative Contract

For normalized image and text representations:

```math
u_i
=
\frac{f_{\mathrm{img}}(x_i)}
{\|f_{\mathrm{img}}(x_i)\|_2},
\qquad
v_i
=
\frac{f_{\mathrm{text}}(t_i)}
{\|f_{\mathrm{text}}(t_i)\|_2},
```

a symmetric contrastive objective is:

```math
\mathcal L_{\mathrm{I\to T}}
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{
    \exp(\gamma u_i^{\mathsf T}v_i)
}{
    \sum_{j=1}^{B}
    \exp(\gamma u_i^{\mathsf T}v_j)
},
```

```math
\mathcal L_{\mathrm{T\to I}}
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{
    \exp(\gamma v_i^{\mathsf T}u_i)
}{
    \sum_{j=1}^{B}
    \exp(\gamma v_i^{\mathsf T}u_j)
}.
```

An autoregressive caption term adds:

```math
\mathcal L_{\mathrm{cap}}
=
-
\sum_{r=1}^{L}
\log
p_{\theta}
\left(
    w_r
    \mid
    w_{<r},
    Z_{\mathrm{img}}
\right).
```

PLIP uses an image-text contrastive contract. CONCH combines alignment with a
separate captioning path. TITAN applies multimodal alignment at ROI and WSI
scales and exports different one-query and 128-query slide objects. These
differences belong in `R`, `H`, and `L`, not merely in a model-name column.

## 9. Knowledge-Conditioned Contract

Let a disease graph be:

```math
\mathcal K
=
\left(
    \mathcal D_{\mathrm{disease}},
    \mathcal E_{\mathrm{hypernym}},
    \mathcal A
\right).
```

The knowledge source changes the target and pair law:

```math
\mathcal Q_{\mathcal K}
\ne
\mathcal Q_{\mathrm{plain}}.
```

For valid-negative indicator `I_ij`:

```math
I_{ij}
=
0
```

can exclude disease-related or hypernym-related groups from negative support.
KEEP therefore changes both semantic geometry and sampling. The graph is not a
posthoc explanation attached after training; it changes which relations enter
the loss.

## 10. Distillation Contract

Let teacher and student output distributions be:

```math
p_t^{(\tau)}(x),
\qquad
p_s^{(\tau)}(x).
```

Output distillation minimizes:

```math
\mathcal L_{\mathrm{out}}
=
\tau^2
D_{\mathrm{KL}}
\left(
    p_t^{(\tau)}
    \,\|\,
    p_s^{(\tau)}
\right).
```

Feature and relation distillation can add:

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

The combined contract is:

```math
\mathcal L_{\mathrm{KD}}
=
\lambda_y\mathcal L_{\mathrm{out}}
+
\lambda_h\mathcal L_{\mathrm{feat}}
+
\lambda_r\mathcal L_{\mathrm{rel}}
+
\lambda_m\mathcal L_{\mathrm{modal}}.
```

Each term preserves a different teacher object. Matching logits does not imply
matching retrieval neighborhoods, spatial context, uncertainty, or multimodal
relations.

## 11. Projection-Head Null Space

Suppose pretraining compares projected features:

```math
z
=
Wh,
\qquad
W\in\mathbb R^{k\times d}.
```

For any vector in the null space:

```math
u
\in
\ker(W),
```

the projected object is unchanged:

```math
W(h+u)
=
Wh.
```

Therefore the pretraining loss can leave null-space components of the exported
backbone feature unconstrained:

```math
\mathcal L(Wh,Q)
=
\mathcal L(W(h+u),Q).
```

This has two consequences:

```text
projection success does not identify every backbone coordinate;
downstream signal can exist in backbone directions not directly organized by
the projection objective.
```

A probe on the exported backbone and retrieval in the projected space evaluate
different geometries.

## 12. Orthogonal Gauge Symmetry

For normalized dot-product objectives, let:

```math
Q^{\mathsf T}Q
=
I.
```

Transform every representation by:

```math
z_i'
=
Qz_i.
```

All pairwise dot products are preserved:

```math
z_i'^{\mathsf T}z_j'
=
z_i^{\mathsf T}Q^{\mathsf T}Qz_j
=
z_i^{\mathsf T}z_j.
```

Thus a contrastive loss cannot identify an absolute coordinate basis. It
identifies relational geometry up to at least orthogonal transformation.

If the downstream reader is retrained, linear accessibility is preserved under
this change. A fixed coordinate-wise interpretation is not.

The meaningful object is therefore an equivalence class:

```math
[Z]_{\mathrm O(d)}
=
\{QZ:Q^{\mathsf T}Q=I\}.
```

## 13. Multiple Global Minima Need Not Preserve The Same Extra Information

Let the contract constrain only statistic `T(X)`. Suppose two encoders are:

```math
\Phi_1(X)
=
\left(
    T(X),
    U(X)
\right),
```

```math
\Phi_2(X)
=
\left(
    T(X),
    0
\right).
```

If the objective reads only `T`, both can attain the same population loss:

```math
\mathcal J_{\mathfrak P}(\Phi_1)
=
\mathcal J_{\mathfrak P}(\Phi_2).
```

Yet a downstream task depending on `U` can distinguish them. Therefore:

```math
\text{same objective value}
\not\Longrightarrow
\text{same downstream sufficiency}.
```

Architecture, optimization, regularization, and finite data decide which
unconstrained information survives.

## 14. Representation Informativeness Partial Order

Let two exported representations of the same source object be:

```math
Z_A
=
\Phi_A(X),
\qquad
Z_B
=
\Phi_B(X).
```

Say `A` refines `B` when there exists a measurable map `T` such that:

```math
\Phi_B
=
T\circ\Phi_A.
```

Write:

```math
\Phi_A
\succeq
\Phi_B.
```

Every downstream rule on `B` can then be reproduced from `A`:

```math
h_B(\Phi_B(X))
=
h_B(T(\Phi_A(X))).
```

For every loss and task distribution, unrestricted Bayes risk satisfies:

```math
\mathcal R^{\star}(Z_A)
\le
\mathcal R^{\star}(Z_B).
```

This is an information order, not a finite-sample performance order. `Z_A` can
be higher dimensional and harder to estimate from a small labeled cohort.

Many foundation representations are incomparable:

```math
\Phi_A
\not\succeq
\Phi_B,
\qquad
\Phi_B
\not\succeq
\Phi_A.
```

For example, a patch-text embedding can preserve semantic directions absent
from a vision-only representation, while a spatial slide token can preserve
co-occurrence absent from the patch-text vector.

There is no task-free scalar ranking in this case.

## 15. Contract Composition Across Scales

For patch, region, and slide stages:

```math
H_i^{(1)}
=
\left\{
    \Phi_1(x_{ij})
\right\}_{j=1}^{n_i},
```

```math
H_i^{(2)}
=
\mathcal R_2
\left(
    \mathcal C_2
    \left(
        H_i^{(1)};
        G_i^{(2)}
    \right)
\right),
```

```math
Z_i
=
\mathcal R_3
\left(
    \mathcal C_3
    \left(
        H_i^{(2)};
        G_i^{(3)}
    \right)
\right).
```

The composed representation is sufficient for a task only if every lower
collision is task-safe:

```math
\Phi_1(x)
=
\Phi_1(x')
\Longrightarrow
\text{no upper stage can distinguish }x\text{ from }x'.
```

HIPT composes patch and region DINO contracts with a slide module.
Prov-GigaPath composes tile DINOv2 with slide masked reconstruction and task
fine-tuning. TITAN composes frozen CONCHv1.5 patch features with slide vision
and language contracts.

The final model name hides the chain of inherited bottlenecks.

## 16. Full Contract Matrix

| Family | Data unit `D` | View or target law `V,Q` | Context `C` | Objective, readout, and export `R,H` | Geometry `G` | Directly constrained statistic |
|---|---|---|---|---|---|---|
| CTransPath | pathology patch with memory support | two augmentations plus mined semantically relevant positives | CNN local field and shifted-window Swin context | MoCo-style projection and SRCL support | normalized angular memory geometry | augmentation-stable patch relation plus mined neighborhood |
| RetCCL | pathology patch, subqueue, and retrieval mosaic | augmented pairs, weighted negatives, cross-branch centroid targets | momentum branches, cluster assignment, and mosaic selection | weighted and group contrast heads; downstream diagnosis-aware query-bag reader | angular, centroid, spatial representative, and archive geometry | cluster-shaped patch relation plus selected retrieval support |
| HIPT | nested 256 and 4096 pretraining objects followed by a WSI task object | DINO global/local views and teacher targets at the 256 and 4096 scales; slide labels enter only at transfer | transformer context within each hierarchy level | patch and region class-token exports followed by a task-trained WSI class token | fixed multiscale partition and token geometry | DINO-stable local and regional summaries plus task-adapted slide composition |
| UNI | broad pathology patch | DINOv2 global/local views and masked-token targets | ViT-L/16 patch context | original class-token patch export | self-distilled patch geometry | broad augmentation-stable patch morphology |
| Virchow | patches sampled in correlated WSI groups | DINOv2 views and token targets | ViT-H/14 patch context | class token concatenated with local-token first moment | self-distilled crop geometry with two exported blocks | patch morphology plus global token and local first moment |
| Phikon-style MIM | patch token sequence | visible tokens predict masked tokens | ViT context over visible tokens | masked-token prediction head | conditional reconstruction geometry | morphology predictable from visible patch context |
| PLIP | OpenPath image-text pair | paired caption and in-batch alternatives | image and text transformers | normalized contrastive vectors | symmetric cosine image-text geometry | caption-visible image semantics |
| CONCH | pathology image-caption pair | contrastive pair plus autoregressive caption | ViT, one-query pooler, 256-query pooler, and text decoder | retrieval vector and caption token set | cosine alignment plus conditional generation | global alignment and caption-predictive image tokens |
| Prov-GigaPath | tile sequence from one WSI | tile DINOv2 plus masked slide-token reconstruction | tile ViT and LongNet dilated slide context | masked reconstruction head and contextual tile or class-token export; downstream ABMIL is separate | quantized coordinates, row-major order, and sparse support | local morphology plus predictable long-context slide structure |
| TITAN | feature-grid ROI or WSI paired with text | iBOT views, synthetic ROI captions, and WSI reports | feature-grid ViT with distance bias and CoCa decoder | one-query retrieval vector and 128-query generation tokens | 2D slide geometry plus image-text space | vision context, ROI semantics, report alignment, and generation state |
| KEEP | semantic image-caption group and disease graph | disease attributes, semantic groups, and hypernym-aware negatives | knowledge, image, and text encoders | group metric head and normalized image-text export; downstream tile-prompt reader is separate | disease hierarchy plus block image-text geometry | knowledge-conditioned disease and image-text relation |
| distillation | student input paired with teacher state | teacher output, feature, relation, or modality target | student and teacher context graphs | projection and distillation heads | teacher-defined output or relation geometry | selected teacher statistic under student capacity |

The final column is deliberately narrow. It names what the objective directly
constrains, not every downstream capability later reported.

## 17. Same Label, Different Contract

The label "self-supervised" contains several non-equivalent target laws:

```math
\begin{aligned}
\text{contrastive}
&:
\quad
\text{relative pair discrimination},
\\
\text{self-distillation}
&:
\quad
\text{teacher distribution agreement},
\\
\text{masked modeling}
&:
\quad
\text{conditional target prediction},
\\
\text{clustering}
&:
\quad
\text{generated partition agreement}.
\end{aligned}
```

Similarly, "multimodal" can mean:

```math
\begin{aligned}
\text{contrastive alignment},
\qquad
\text{autoregressive generation},
\qquad
\text{knowledge-conditioned relation},
\qquad
\text{retrieval through a memory}.
\end{aligned}
```

These objectives can coexist in one model, but they do not become one
mathematical operation.

## 18. Minimal Contract Comparison

Before comparing two foundation models, hold or report:

```text
source object and independent sampling unit;
patch, region, slide, report, and patient weighting;
view, crop, mask, and coordinate law;
positive, negative, teacher, caption, report, or knowledge relation;
context support before the objective;
pretraining readout and projection head;
exported checkpoint object and tensor shape;
loss weights, temperature, and normalization;
comparison metric and memory support;
downstream reader and trainable parameter set.
```

If these differ, the experiment compares contracts. That may be scientifically
appropriate, but the interpretation should not attribute the entire difference
to model scale or backbone architecture.

## 19. C/R/G/S Placement

The contract itself can be placed in the repo-wide decomposition:

```math
\widetilde H
=
\mathcal C
\left(
    H;
    G,
    S
\right),
```

```math
U
=
\mathcal R
\left(
    \widetilde H
\right),
```

```math
\widehat Q
=
\mathcal H
\left(
    U
\right).
```

Here:

```text
C is the architecture and interaction support;
R is the token, moment, class-token, query, or masked-position readout;
G is the augmentation, spatial, metric, hierarchy, or modality geometry;
S is the generated or observed pretraining supervision;
H is the objective-specific prediction or projection head;
U is the surviving pretraining statistic.
```

The scalar loss evaluates `H(U)` against `S`; it does not make `U` uniquely
identifiable.

## 20. Bottom Line

A pathology foundation model is mathematically described by:

```math
\boxed{
\text{sampling law}
+
\text{view law}
+
\text{target relation}
+
\text{context}
+
\text{readout}
+
\text{head}
+
\text{loss}
+
\text{geometry}.
}
```

The pretraining contract determines which relations receive direct pressure.
It does not uniquely determine every backbone coordinate, every downstream
statistic, or every global minimum.

The correct comparison is not:

```math
\text{Which foundation model is largest?}
```

It is:

```math
\boxed{
\text{Which statistical contract was optimized, which object was exported,
which distinctions are identified away, and which representations remain
incomparable without naming a downstream task?}
}
```
