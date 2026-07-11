# Cross-Slide Statistical Objects

Primary anchors:

- Wang et al. "SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised
  Whole-Slide Image Classification." NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/726204cea3ec27790a644e5b379175e3-Abstract-Conference.html
- Juyal et al. "SC-MIL: Supervised Contrastive Multiple Instance Learning for
  Imbalanced Classification in Digital Pathology." WACV 2024.
  https://openaccess.thecvf.com/content/WACV2024/html/Juyal_SC-MIL_Supervised_Contrastive_Multiple_Instance_Learning_for_Imbalanced_Classification_in_WACV_2024_paper.html

## Slide As A Labeled Bag

For slide `i`:

```math
\mathcal{X}_i
=
\left\{
x_{i1},\ldots,x_{in_i}
\right\},
\qquad
Y_i
\in
\left\{
1,ldots,C
\right\}.
```

Patch labels are latent:

```math
y_{ij}
\quad
\text{unobserved}.
```

Under binary standard MIL:

```math
Y_i
=
\max_j y_{ij}.
```

Same slide label therefore does not imply identical patch distributions.

## Three Cross-Slide Units

Whole-bag embedding:

```math
b_i
=
\mathcal{R}
\left(
\left\{
f(x_{ij})
\right\}_{j=1}^{n_i}
\right).
```

Selected subbag:

```math
\mathcal{B}_i^{(k)}
=
\left\{
f(x_{ij}):
j\in I_i^{(k)}
\right\}.
```

Mixed bag:

```math
\widetilde{\mathcal{X}}_{ab}
=
\mathcal{M}
\left(
\mathcal{X}_a,
\mathcal{X}_b
\right).
```

SC-MIL contrasts `b_i`. SCL-WC contrasts patches against selected subbags.
CAMCSA creates mixed bags but does not define a contrastive denominator.

## Label Relation Versus Morphology Relation

Bag-level supervised contrast uses:

```math
R_{ij}^{Y}
=
\mathbb{1}
\left\{
Y_i=Y_j
\right\}.
```

A latent morphology relation is:

```math
R_{ij}^{\mu}
=
\mathbb{1}
\left\{
\mu_i
\sim
\mu_j
\right\},
```

where the latent measure is the slide's patch distribution. In general:

```math
R_{ij}^{Y}
\ne
R_{ij}^{\mu}.
```

Two cancer-positive slides can contain different subtypes, grades, tumor
fractions, and microenvironments.

## Two Levels Of False Negatives

Slide-level false negative:

```math
Y_i\ne Y_j,
\qquad
\mu_i\sim\mu_j.
```

Patch-level false negative inside different labeled slides:

```math
Y_i\ne Y_j,
\qquad
y_{ia}=y_{jb}=0.
```

Normal tissue is shared across many disease labels. Cross-slide contrast can
repel common benign morphology unless patch selection removes it.

## WSI Contrast Composition

```math
\text{slide label}
\longrightarrow
\text{attention or bag readout}
\longrightarrow
\text{positive relation}
\longrightarrow
\text{contrastive geometry}.
```

Unlike self-supervised patch contrast, cross-slide contrast inherits the weak
label and the aggregation operator before defining positives.
