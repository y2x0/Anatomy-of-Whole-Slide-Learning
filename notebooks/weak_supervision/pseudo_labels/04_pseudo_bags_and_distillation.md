# Pseudo-Bags And Distillation

Pseudo-bag methods reduce a huge slide bag into smaller bags, train a local MIL
model on those smaller bags, distill features from that model, and then train a
second slide-level MIL model.

DTFD-MIL is the anchor example. It is not a generic teacher-student method. Its
specific move is double-tier feature distillation.

References: [DTFD-MIL arXiv](https://arxiv.org/abs/2203.12081),
[CVPR open-access paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_paper.pdf).

## Tier-1 Pseudo-Bag Partition

For slide $i$, randomly partition instances into $M_i$ pseudo-bags:

```math
\{1,\ldots,n_i\}
=
\bigcup_{m=1}^{M_i}
\mathcal{B}_{im},
\qquad
\mathcal{B}_{im}\cap\mathcal{B}_{im'}=\varnothing.
```

Each pseudo-bag inherits the parent slide label:

```math
\widehat Y_{im}
=
Y_i.
```

This is a generated target:

```math
\widehat Y_{im}
=
\Psi_{\mathrm{split}}(Y_i,\mathcal{B}_{im}),
```

not an observed pseudo-bag label.

## Tier-1 MIL

The first-tier model aggregates each pseudo-bag:

```math
z_{im}^{(1)}
=
\mathcal{R}_1(\{h_{ij}:j\in\mathcal{B}_{im}\}),
```

and predicts:

```math
\widehat p_{im}^{(1)}
=
\mathcal{H}_1(z_{im}^{(1)}).
```

The Tier-1 loss copies the slide label to the pseudo-bag:

```math
\mathcal{L}_1
=
\sum_{i,m}
\mathrm{CE}(\widehat p_{im}^{(1)},Y_i).
```

This is noisy. If $Y_i=1$, many pseudo-bags can contain no positive evidence:

```math
Y_i=1,
\qquad
Y_{im}^{\star}=0,
\qquad
\widehat Y_{im}=1.
```

DTFD-MIL accepts that noise because pseudo-bags are much smaller than whole
slides and can yield more direct optimization signals.

## Why Attention Is Not Treated As Probability

Under attention MIL, a pseudo-bag representation may be:

```math
z_{im}^{(1)}
=
\sum_{j\in\mathcal{B}_{im}}
a_{ij}^{(m)}v(h_{ij}).
```

The attention weight $a_{ij}^{(m)}$ is a readout coefficient. It is not, by
itself, an instance probability:

```math
a_{ij}^{(m)}
\ne
P(Z_{ij}=1\mid H_i,Y_i)
```

unless an additional probabilistic model justifies that interpretation.

DTFD-MIL therefore uses gradient-based evidence, described in the paper through
Grad-CAM, to derive an instance-discriminative score from the Tier-1 model.
Schematically:

```math
\rho_{ij}^{(m)}
=
\mathrm{Normalize}
\left(
\mathrm{GradCAM}
(o_{im}^{(1)},h_{ij})
\right).
```

The important point is not the exact notation above; it is the distinction:
derived instance evidence is computed from the trained pseudo-bag classifier,
rather than read directly from raw attention weights.

## Feature Distillation

Tier-1 produces a pseudo-bag feature set:

```math
\{h_{ij}:j\in\mathcal{B}_{im}\}
\longrightarrow
u_{im}.
```

DTFD-MIL compares feature distillation strategies that choose or aggregate
features using derived instance evidence or attention-like scores:

```text
MaxS:
    select the feature with maximum derived instance score

MaxMinS:
    use both maximum-score and minimum-score evidence

MAS:
    select by maximum attention score

AFS:
    aggregate or select features with a learned/attention-based feature rule
```

Mathematically, these strategies define a second generated target:

```math
u_{im}
=
\Phi_{\mathrm{distill}}
\left(
\{h_{ij}\}_{j\in\mathcal{B}_{im}},
\{\rho_{ij}^{(m)}\}_{j\in\mathcal{B}_{im}},
\{a_{ij}^{(m)}\}_{j\in\mathcal{B}_{im}}
\right).
```

The feature $u_{im}$ is not an observed region label. It is an algorithmic
summary chosen by the Tier-1 model.

## Tier-2 Slide MIL

The second tier treats distilled pseudo-bag features as instances:

```math
z_i^{(2)}
=
\mathcal{R}_2(\{u_{im}\}_{m=1}^{M_i}),
```

```math
\widehat p_i^{(2)}
=
\mathcal{H}_2(z_i^{(2)}).
```

The slide loss is:

```math
\mathcal{L}_2
=
\sum_i
\mathrm{CE}(\widehat p_i^{(2)},Y_i).
```

The full logic is therefore:

```math
Y_i
\to
\{\widehat Y_{im}\}_m
\to
\{u_{im}\}_m
\to
\widehat Y_i.
```

## Bias-Variance Tradeoff

Pseudo-bags reduce bag size:

```math
|\mathcal{B}_{im}|
\ll
n_i.
```

This improves optimization and can concentrate evidence. But it changes the
noise model from slide-label noise:

```math
P(\widetilde Y_i\ne Y_i)
```

to pseudo-bag label noise:

```math
P(\widehat Y_{im}\ne Y_{im}^{\star}\mid Y_i).
```

Positive slides are the dangerous case because copied positive labels create
many false-positive pseudo-bags.

## C/R/G/S Placement

```text
G:
    pseudo-bags are random partitions in DTFD-MIL, though other methods may use spatial or clustered splits

C:
    Tier-1 local MIL shapes pseudo-bag features before Tier-2 slide MIL

R:
    local readout creates distilled pseudo-bag features; global readout aggregates them

S:
    observed slide labels are copied to pseudo-bags and converted into distilled features
```

## Dense Summary

DTFD-MIL trades one weakly supervised problem:

```math
Y_i
\text{ supervises }
\{h_{ij}\}_{j=1}^{n_i}
```

for a two-stage generated-target chain:

```math
Y_i
\to
\widehat Y_{im}
\to
u_{im}
\to
\widehat Y_i.
```

The method succeeds when smaller pseudo-bags and distilled features improve
optimization more than copied slide labels corrupt local learning.
