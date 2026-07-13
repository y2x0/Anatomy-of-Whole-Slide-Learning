# Paper Placement Matrix

This file places anchor papers by supervision channel. The point is not to rank
methods. The point is to avoid a common mistake:

```text
architecture name != supervision model
```

A method can use slide labels, report labels, pseudo-instance constraints,
contrastive pairs, teacher targets, and copied pseudo-bag labels in the same
training pipeline. Each one has a different mathematical status.

## Compact Matrix

| Method | Observed S^{\mathrm{obs}} | Generated target | Core mechanism | Identifiable from loss | Main non-identifiability |
|---|---|---|---|---|---|
| ABMIL | slide label Y_i | none | attention-weighted bag embedding | slide prediction function | attention as instance truth |
| Campanella-style weak supervision | report/case-derived label \widetilde Y | slide/case mapping target | large-scale slide CE and aggregation | clinical-label predictor | slide-local diagnostic region |
| CLAM | slide label Y_i | top/bottom-k attention constraints | slide CE plus smooth-SVM instance clustering | slide class and selected-extreme geometry | whether selected patches are true witnesses |
| DSMIL | slide label Y_i | critical-instance index induced by max branch | dual-stream max witness plus bag relation branch | slide predictor with critical-instance bias | true instance labels |
| DTFD-MIL | slide label Y_i | pseudo-bag labels and distilled features | Tier-1 pseudo-bag MIL, Grad-CAM evidence, Tier-2 MIL | slide predictor over distilled pseudo-bags | true pseudo-bag labels |
| Cluster-to-Conquer | slide label Y_i | local clusters and copied patch labels | k-means sampling, patch CE, slide CE, KL attention regularization | slide predictor under cluster sampling | cluster diagnostic meaning |
| SC-MIL | slide labels | same-label positive pairs | supervised contrastive slide embeddings plus CE | label-discriminative slide geometry | same-label morphology equivalence |
| SCL-WC | slide labels plus selected cross-slide relations | weakly consistent positives/negatives | self-supervised features plus weak contrastive refinement | relation-conditioned representation | correctness of cross-slide relation |
| LACL | labels plus lesion-aware pair construction | lesion-filtered queues/pairs | lesion-aware contrastive objective | geometry under lesion heuristic | true lesion equivalence |
| RetCCL / CTransPath | unlabeled images and augmentations/clusters | positive/negative SSL pairs | pathology contrastive pretraining | invariant patch/slide embedding | downstream task relevance |
| CONCH / PLIP | image-text pairs | paired cross-modal index labels | image-text contrastive alignment | text-aligned image embedding | local visual truth behind text |

## ABMIL

Reference: [Attention-based Deep MIL](https://arxiv.org/abs/1802.04712).

ABMIL observes only:

```math
S_i^{\mathrm{obs}}
=
Y_i.
```

The slide representation is:

```math
z_i
=
\sum_{j=1}^{n_i}
a_{ij}v(h_{ij}),
```

with:

```math
a_{ij}
=
\frac{\exp s(h_{ij})}
{\sum_{\ell=1}^{n_i}\exp s(h_{i\ell})}.
```

The loss constrains:

```math
\mathcal{H}(z_i)
\approx
Y_i.
```

It does not constrain:

```math
a_{ij}
\approx
P(Z_{ij}=1\mid H_i,Y_i).
```

So the identifiable object is the slide predictor, not an instance map.

## Campanella-Style Clinical Weak Supervision

Reference: [Clinical-scale weak supervision for WSI](https://pmc.ncbi.nlm.nih.gov/articles/PMC7418463/).

The observed label is extracted from clinical metadata or reports:

```math
\widetilde Y_{\mathrm{case}}
=
E_{\mathrm{report}}(R_{\mathrm{case}}).
```

Then a slide inherits a case-level or specimen-level target through a mapping:

```math
\widetilde Y_{i}
=
M_{\mathrm{case\to slide}}(\widetilde Y_{\mathrm{case}},i).
```

The actual channel is:

```math
U_{\mathrm{case}}
\to
R_{\mathrm{case}}
\to
\widetilde Y_{\mathrm{case}}
\to
\widetilde Y_i.
```

The loss is usually slide-level CE:

```math
\mathcal{L}
=
\sum_i
\mathrm{CE}(\widehat y_i,\widetilde Y_i).
```

The method can learn a strong clinical-label predictor at scale. But the label
does not automatically identify which slide, region, or patch expresses the
case-level diagnosis.

## CLAM

Reference: [CLAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC8711640/).

CLAM observes:

```math
S_i^{\mathrm{obs}}
=
Y_i.
```

It generates top and bottom attention sets:

```math
(\mathcal{T}_i^+(c),\mathcal{T}_i^-(c))
=
\Psi_{\mathrm{CLAM}}(A_i,Y_i).
```

For
```math
Y_i=c
```
:

```math
\mathcal{T}_i^+(c)
=
\mathrm{TopK}_{j}\ a_{ij}^{(c)},
\qquad
\mathcal{T}_i^-(c)
=
\mathrm{BottomK}_{j}\ a_{ij}^{(c)}.
```

The auxiliary target is:

```math
\widehat Z_{ij}^{(c)}
=
1
\quad
j\in\mathcal{T}_i^+(c),
```

```math
\widehat Z_{ij}^{(c)}
=
0
\quad
j\in\mathcal{T}_i^-(c).
```

The paper uses a binary smooth-SVM-style clustering loss on selected instances:

```math
\mathcal{L}_{\mathrm{inst}}
=
\frac{1}{2K}
\left[
\sum_{j\in\mathcal{T}_i^+(c)}
\ell_{\mathrm{svm}}(r_c(h_{ij}),1)
+
\sum_{j\in\mathcal{T}_i^-(c)}
\ell_{\mathrm{svm}}(r_c(h_{ij}),0)
\right].
```

This is not ordinary instance cross-entropy and not literal EM unless a
likelihood is defined for which top/bottom-k selection is a MAP step.

## DSMIL

References: [DSMIL arXiv](https://arxiv.org/abs/2011.08939),
[DSMIL PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8765709/).

DSMIL observes slide labels:

```math
S_i^{\mathrm{obs}}
=
Y_i.
```

The instance stream scores each patch:

```math
s_{ijc}
=
g_c(h_{ij}).
```

The critical instance for class
```math
c
```
is induced by a max operation:

```math
j_i^\star(c)
=
\arg\max_j s_{ijc}.
```

The bag stream uses the critical instance as a query-like reference for
relations to other instances:

```math
a_{ij}^{(c)}
\propto
\exp
\left(
q(h_{i,j_i^\star(c)})^\top k(h_{ij})
\right).
```

The supervision is not self-supervised contrastive learning itself. Contrastive
pretraining can provide better patch embeddings, but DSMIL's weak-supervision
core is the dual-stream MIL aggregator under slide labels.

The identifiable object is:

```math
H_i
\mapsto
\widehat Y_i,
```

with a critical-instance inductive bias. The critical instance is not an
observed patch label.

## DTFD-MIL

References: [DTFD-MIL arXiv](https://arxiv.org/abs/2203.12081),
[CVPR open-access paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_paper.pdf).

DTFD-MIL observes:

```math
S_i^{\mathrm{obs}}
=
Y_i.
```

It randomly partitions a slide into pseudo-bags:

```math
\{1,\ldots,n_i\}
=
\bigcup_{m=1}^{M_i}
\mathcal{B}_{im}.
```

Then it copies the parent slide label:

```math
\widehat Y_{im}
=
Y_i.
```

Tier-1 MIL learns pseudo-bag predictors:

```math
z_{im}^{(1)}
=
\mathcal{R}_1(\{h_{ij}:j\in\mathcal{B}_{im}\}),
\qquad
\widehat p_{im}^{(1)}
=
\mathcal{H}_1(z_{im}^{(1)}).
```

Because attention is not automatically an instance probability, DTFD-MIL uses
Grad-CAM-style evidence from the pseudo-bag classifier to guide feature
distillation:

```math
\rho_{ij}^{(m)}
=
\mathrm{Normalize}
\left(
\mathrm{GradCAM}(o_{im}^{(1)},h_{ij})
\right).
```

The distilled pseudo-bag features become Tier-2 instances:

```math
z_i^{(2)}
=
\mathcal{R}_2(\{u_{im}\}_{m=1}^{M_i}).
```

The main failure is copied pseudo-bag label noise:

```math
Y_i=1,
\qquad
Y_{im}^{\star}=0,
\qquad
\widehat Y_{im}=1.
```

## Cluster-To-Conquer

Reference: [Cluster-to-Conquer](https://proceedings.mlr.press/v143/sharma21a.html).

Cluster-to-Conquer observes only slide labels:

```math
S_i^{\mathrm{obs}}
=
Y_i.
```

It creates local slide clusters:

```math
\kappa_i(j)
\in
\{1,\ldots,K_i\},
\qquad
\mathcal{C}_{ik}
=
\{j:\kappa_i(j)=k\}.
```

The generated patch label is copied from the slide:

```math
\widehat Z_{ij}
=
Y_i.
```

The loss has three supervision shapes:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{slide}}
+
\lambda_{\mathrm{patch}}\mathcal{L}_{\mathrm{patch}}
+
\lambda_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}}.
```

The KL term regularizes attention within local clusters toward a less collapsed
distribution:

```math
\mathcal{L}_{\mathrm{KL}}
=
\sum_{i,k}
\mathrm{KL}
\left(
u_i^{(k)}
\;\|\;
\bar a_i^{(k)}
\right).
```

The generated cluster structure is not a proof of diagnostic equivalence:

```math
h_{ij},h_{i\ell}\in\mathcal{C}_{ik}
\not\Rightarrow
Z_{ij}=Z_{i\ell}.
```

## SC-MIL

Reference: [SC-MIL](https://openaccess.thecvf.com/content/WACV2024/papers/Juyal_SC-MIL_Supervised_Contrastive_Multiple_Instance_Learning_for_Imbalanced_Classification_in_WACV_2024_paper.pdf).

SC-MIL uses slide labels to define supervised contrastive positives between
slide embeddings:

```math
\mathcal{P}(i)
=
\{p\ne i:Y_p=Y_i\}.
```

The contrastive loss shapes the slide representation before or alongside
classification. SC-MIL uses a training curriculum:

```math
\mathcal{L}_{\mathrm{SC-MIL}}(t)
=
\beta_t
\mathcal{L}_{\mathrm{supcon}}
+
(1-\beta_t)
\mathcal{L}_{\mathrm{CE}},
```

where
```math
\beta_t
```
decays so training progressively shifts from bag-level
representation learning to classifier learning.

The assumption is:

```math
Y_p=Y_i
\Rightarrow
U_p\sim U_i.
```

This can fail when a class contains multiple morphologic modes or when imbalance
creates repeated majority-class negatives.

## SCL-WC

Reference: [SCL-WC](https://proceedings.neurips.cc/paper_files/paper/2022/hash/726204cea3ec27790a644e5b379175e3-Abstract-Conference.html),
[SCL-WC PDF](https://openreview.net/pdf?id=1fKJLRTUdo).

SCL-WC combines self-supervised feature learning with weakly supervised
contrastive refinement. Its task-specific aggregation has three named pieces:

```text
CDA:
    class-specific deep attention for WSI prediction

PNM:
    positive-negative-aware modeling inside each WSI

WSCL:
    weakly supervised cross-slide contrastive learning
```

CDA creates class-specific slide features:

```math
\widetilde f_i^{(c)}
=
\sum_j a_{ij}^{(c)}f_{ij}.
```

PNM adds intra-slide positive/negative separation pressure from attention and
weak labels:

```math
\mathcal{L}_{\mathrm{PNM}}
=
\ell_{\mathrm{pos}}(\mathcal{F}_i^+)
+
\ell_{\mathrm{neg}}(\mathcal{F}_i^-).
```

WSCL defines the cross-slide relation:

```math
(i,p)
\in
\mathcal{P}_{\mathrm{WSCL}}
\quad
\text{when }
Y_i=Y_p.
```

The failure mode is both pair construction and attention-set construction:

```math
(i,p)\in\mathcal{P}_{\mathrm{WSCL}}
\not\Rightarrow
U_i\sim U_p.
```

## LACL

Reference: [LACL](https://arxiv.org/abs/2206.13115).

LACL constructs lesion-aware positive and negative sets:

```math
\mathcal{P}_{\mathrm{LACL}}(i),
\qquad
\mathcal{N}_{\mathrm{LACL}}(i).
```

The contrastive objective is standard in form:

```math
-
\log
\frac{
\sum_{p\in\mathcal{P}_{\mathrm{LACL}}(i)}
\exp(z_i^\top z_p/\tau)
}{
\sum_{p\in\mathcal{P}_{\mathrm{LACL}}(i)}
\exp(z_i^\top z_p/\tau)
+
\sum_{n\in\mathcal{N}_{\mathrm{LACL}}(i)}
\exp(z_i^\top z_n/\tau)
}.
```

The novelty is the construction of
```math
\mathcal{P}
```
 and
```math
\mathcal{N}
```
, not the
softmax. The method assumes lesion-aware selection reduces class collision:

```math
P(U_n\sim U_i\mid n\in\mathcal{N}_{\mathrm{LACL}})
<
P(U_n\sim U_i\mid n\sim P_K).
```

## RetCCL And CTransPath

References: [RetCCL](https://pubmed.ncbi.nlm.nih.gov/36270093/),
[CTransPath](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043).

These are representation-learning papers used upstream of MIL. The observed
supervision is not a diagnostic label:

```math
S^{\mathrm{obs}}
=
\{(t_1(x),t_2(x))\text{ are linked views or mined positives}\}.
```

The representation is learned by invariance and contrast:

```math
f(t_1(x))
\approx
f(t_2(x)),
\qquad
f(x)
\not\approx
f(x^-).
```

For downstream WSI learning, the learned encoder supplies:

```math
x_{ij}
\xrightarrow{f_{\mathrm{pre}}}
h_{ij}
\xrightarrow{\mathrm{MIL}}
\widehat y_i.
```

The weak supervision is imported geometry. It may or may not align with the
downstream label geometry.

## CONCH And PLIP

References: [CONCH](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/),
[PLIP](https://www.nature.com/articles/s41591-023-02504-3).

Vision-language pathology models observe image-text pairs:

```math
S^{\mathrm{obs}}
=
\{(x_i,t_i)\}.
```

The contrastive target is an index relation:

```math
t_i
\text{ is the paired text for }
x_i.
```

The image-text contrastive loss is:

```math
\mathcal{L}_{\mathrm{ITC}}
=
-
\log
\frac{
\exp(f_I(x_i)^\top f_T(t_i)/\tau)
}{
\sum_m
\exp(f_I(x_i)^\top f_T(t_m)/\tau)
}.
```

This supervision can encode richer semantics than a class label. But it does
not guarantee that every local visual feature in
```math
x_i
```
 is described by
```math
t_i
```
:

```math
(x_i,t_i)\text{ paired}
\not\Rightarrow
t_i\text{ labels every patch in }x_i.
```

## Dense Summary

For each paper, ask four questions:

```text
1. What was observed from the world?
2. What did the algorithm generate?
3. Which loss treats the generated object as a target?
4. Which latent variable is still not identified?
```

That is the weak-supervision map. The same architecture can move categories if
the supervision channel changes.
