# WSI Contrastive Method Cards

Contrastive pathology papers differ less by architecture than by the relation
they decide to call positive. The mathematical object is:

```math
S^{\mathrm{obs}}
=
\mathcal{P}
\cup
\mathcal{N},
```

where $\mathcal{P}$ is a set of pairs or groups to pull together, and
$\mathcal{N}$ is a sampled set to push apart.

## SC-MIL

SC-MIL uses slide labels to define supervised contrastive relations between
bag embeddings.

Reference: [SC-MIL paper](https://openaccess.thecvf.com/content/WACV2024/papers/Juyal_SC-MIL_Supervised_Contrastive_Multiple_Instance_Learning_for_Imbalanced_Classification_in_WACV_2024_paper.pdf).

Let $z_i$ be a slide embedding from a MIL encoder. The positive set for anchor
$i$ is:

```math
\mathcal{P}(i)
=
\{p\ne i:Y_p=Y_i\}.
```

A supervised contrastive term has the form:

```math
\mathcal{L}_{\mathrm{supcon}}(i)
=
-
\frac{1}{|\mathcal{P}(i)|}
\sum_{p\in\mathcal{P}(i)}
\log
\frac{
\exp(z_i^\top z_p/\tau)
}{
\sum_{a\ne i}\exp(z_i^\top z_a/\tau)
}.
```

The slide classifier still uses cross-entropy:

```math
\mathcal{L}_{\mathrm{CE}}
=
\sum_i
\mathrm{CE}(\widehat y_i,Y_i).
```

SC-MIL combines the two with a training-time curriculum:

```math
\mathcal{L}_{\mathrm{SC-MIL}}(t)
=
\beta_t
\mathcal{L}_{\mathrm{supcon}}
+
(1-\beta_t)
\mathcal{L}_{\mathrm{CE}},
```

where $\beta_t$ decays over training. Early updates emphasize balanced
bag-level feature geometry; later updates emphasize classifier fitting.

The method's supervision assumption is:

```math
Y_p=Y_i
\Rightarrow
U_p\sim U_i.
```

In imbalanced WSI classification, this can improve representation geometry, but
it can also over-collapse heterogeneous same-label slides.

## SCL-WC

SCL-WC separates generic self-supervised feature learning from weakly
supervised WSI refinement. The paper's structure is not just "same class
slides are positives"; it combines task-agnostic feature extraction with three
task-specific aggregation modules.

Reference: [SCL-WC abstract](https://proceedings.neurips.cc/paper_files/paper/2022/hash/726204cea3ec27790a644e5b379175e3-Abstract-Conference.html),
[SCL-WC PDF](https://openreview.net/pdf?id=1fKJLRTUdo).

A compact decomposition is:

```math
x
\xrightarrow{\mathrm{SSL}}
h
\xrightarrow{\mathrm{CDA},\mathrm{PNM},\mathrm{WSCL}}
\widetilde h
\xrightarrow{\mathrm{MIL}}
\widehat y.
```

Class-specific deep attention (CDA) creates class-specific slide features:

```math
\widetilde f_i^{(c)}
=
\sum_j a_{ij}^{(c)}f_{ij}.
```

Positive-negative-aware modeling (PNM) adds an intra-WSI separation pressure
between high-attention tumor-like patches and normal-like patches. In abstract
form:

```math
\mathcal{L}_{\mathrm{PNM}}
=
\ell_{\mathrm{pos}}(\mathcal{F}_i^+)
+
\ell_{\mathrm{neg}}(\mathcal{F}_i^-),
```

where the positive/negative sets are induced by attention and slide labels, not
by true patch annotations.

Weakly supervised cross-slide contrastive learning (WSCL) creates inter-WSI
relations:

```math
\mathcal{P}_{\mathrm{WSCL}}(i)
=
\{p\ne i:Y_p=Y_i\},
```

and pushes different slide classes away. The total task-specific aggregator is
therefore closer to:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{CDA}}
+
\lambda_{\mathrm{PNM}}\mathcal{L}_{\mathrm{PNM}}
+
\lambda_{\mathrm{WSCL}}\mathcal{L}_{\mathrm{WSCL}}.
```

The mathematical risk is relation and attention-set misspecification:

```math
(i,j)\in\mathcal{P}_{\mathrm{WSCL}}
\not\Rightarrow
U_i\sim U_j.
```

This is especially delicate in pathology because slides with the same class can
share label but not morphology, and slides with different labels can share
background tissue.

## LACL

LACL is lesion-aware contrastive learning. Its central move is to reduce false
negative/false positive pairs by constructing lesion-aware queues or memory
sets rather than using naive batch negatives.

Reference: [LACL paper](https://arxiv.org/abs/2206.13115).

Let $q_i$ be an anchor embedding and let $\mathcal{Q}_c$ be a class-specific or
lesion-refined memory queue. A contrastive denominator may be restricted to:

```math
\mathcal{N}_{\mathrm{LACL}}(i)
=
\{k\in\mathcal{Q}:k\text{ passes lesion-aware negative filtering}\}.
```

The positive set similarly uses lesion-aware selection:

```math
\mathcal{P}_{\mathrm{LACL}}(i)
=
\{p:p\text{ is selected as lesion-consistent with }i\}.
```

The core mathematical claim is:

```math
P(U_p\sim U_i\mid p\in\mathcal{P}_{\mathrm{LACL}})
>
P(U_p\sim U_i\mid Y_p=Y_i).
```

and:

```math
P(U_k\sim U_i\mid k\in\mathcal{N}_{\mathrm{LACL}})
<
P(U_k\sim U_i\mid k\sim P_K).
```

Thus LACL is primarily a pair-construction method. Its success depends on the
lesion heuristic being more faithful than naive class or batch sampling.

## RetCCL And CTransPath

RetCCL and CTransPath are contrastive or self-supervised pathology
representation learners used before downstream WSI MIL.

References: [RetCCL](https://pubmed.ncbi.nlm.nih.gov/36270093/),
[CTransPath](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043).

The observed signal is not a diagnosis:

```math
S^{\mathrm{obs}}
=
\{(t_1(x),t_2(x))\text{ are two views of the same source}\}.
```

The latent assumption is augmentation invariance:

```math
U(t_1(x))
=
U(t_2(x)).
```

For downstream WSI learning, these models inject task information indirectly:

```math
x_{ij}
\xrightarrow{f_{\mathrm{pre}}}
h_{ij}
\xrightarrow{\mathrm{MIL}}
\widehat y_i.
```

The risk is that the pretraining geometry may preserve stain, scanner, organ,
or tissue similarity more strongly than the downstream label geometry.

## CONCH And PLIP

CONCH and PLIP use image-text or image-report pairs. The weak supervision is a
cross-modal relation:

References: [CONCH](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/),
[PLIP](https://www.nature.com/articles/s41591-023-02504-3).

```math
S^{\mathrm{obs}}
=
\{(x,t):x\text{ is paired with text }t\}.
```

The contrastive objective aligns image and text embeddings:

```math
\mathcal{L}_{\mathrm{ITC}}
=
-
\log
\frac{
\exp(f_I(x_i)^\top f_T(t_i)/\tau)
}{
\sum_{m}
\exp(f_I(x_i)^\top f_T(t_m)/\tau)
}.
```

The latent assumption is not simply class equality. It is semantic agreement:

```math
t_i
\text{ describes the pathology-relevant content of }
x_i.
```

This can be much richer than slide labels, but it imports language noise:
report conventions, missing negatives, diagnostic uncertainty, and mismatch
between local image crops and global clinical text.

## Dense Summary

Contrastive WSI methods should always state:

```text
positive construction:
    what makes two objects equivalent?

negative construction:
    what makes two objects different?

object level:
    patch, region, slide, patient, report, or memory prototype?

failure mode:
    which latent morphologies are incorrectly pulled together or pushed apart?
```

The loss is not the hard part. The hard part is whether the pair relation is
biologically and diagnostically meaningful.
