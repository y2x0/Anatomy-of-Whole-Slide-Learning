# Patch Encoder To WSI Predictor Interface

A pretrained patch encoder defines the alphabet from which a slide learner is
allowed to build its prediction. When the encoder is frozen, information
discarded before aggregation cannot be recovered by a stronger MIL head.

Primary anchors:

- DSMIL: https://arxiv.org/abs/2011.08939
- CTransPath: https://pubmed.ncbi.nlm.nih.gov/35952419/
- CLAM, used as the CTransPath WSI aggregator:
  https://arxiv.org/abs/2004.09666

## Two-Stage Composition

For slide `i`:

```math
\mathcal{B}_i
=
\left\{
x_{ij}
\right\}_{j=1}^{n_i}.
```

A pretrained encoder produces:

```math
h_{ij}
=
f_{\widehat\theta}
\left(
x_{ij}
\right)
\in
\mathbb{R}^{d}.
```

A slide learner produces:

```math
\widehat y_i
=
\mathcal{M}_{\psi}
\left(
\left\{
h_{ij}
\right\}_{j=1}^{n_i}
\right).
```

The complete map is:

```math
\widehat y_i
=
\left(
\mathcal{M}_{\psi}
\circ
f_{\widehat\theta}^{\otimes n_i}
\right)
\left(
\mathcal{B}_i
\right).
```

## Encoder Pushforward Measure

Represent the empirical patch distribution of slide `i` by:

```math
\widehat\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\delta_{x_{ij}}.
```

The encoder induces the pushforward:

```math
\widehat\nu_i
=
\left(
f_{\widehat\theta}
\right)_{\#}
\widehat\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\delta_{h_{ij}}.
```

A general permutation-invariant slide model using no coordinates can be
written as:

```math
\widehat y_i
=
T_{\psi}
\left(
\widehat\nu_i,
n_i
\right).
```

Cardinality-blind mean- or attention-normalized models reduce further to:

```math
\widehat y_i
=
T_{\psi}
\left(
\widehat\nu_i
\right).
```

The raw image distribution `\widehat\mu_i` has been replaced by the
feature distribution `\widehat\nu_i`, with `n_i` available only if
the downstream design retains it.

## Encoder Equivalence Classes

Define:

```math
x
\sim_f
x'
\quad
\Longleftrightarrow
\quad
f_{\widehat\theta}(x)
=
f_{\widehat\theta}(x').
```

Every downstream learner that only sees frozen features is constant on these
equivalence classes. If:

```math
x_{\mathrm{rare}}
\sim_f
x_{\mathrm{common}},
```

then the slide head cannot distinguish the rare and common patches from their
features.

## Bag-Level Indistinguishability

Two slides are encoder-indistinguishable as sets if:

```math
\left\{
f_{\widehat\theta}(x_{aj})
\right\}_{j=1}^{n_a}
=
\left\{
f_{\widehat\theta}(x_{bj})
\right\}_{j=1}^{n_b}
```

as multisets.

For any deterministic set aggregator:

```math
\mathcal{M}_{\psi}
\left(
\left\{
f_{\widehat\theta}(x_{aj})
\right\}
\right)
=
\mathcal{M}_{\psi}
\left(
\left\{
f_{\widehat\theta}(x_{bj})
\right\}
\right).
```

No attention, transformer, or prototype readout can separate identical input
multisets.

## Geometry Can Be Lost Twice

Let each patch also have coordinate `c_{ij}`. The raw slide object is:

```math
\left\{
\left(
x_{ij},c_{ij}
\right)
\right\}_{j=1}^{n_i}.
```

Patch-only encoding gives:

```math
\left\{
\left(
h_{ij},c_{ij}
\right)
\right\}_{j=1}^{n_i}.
```

Dropping coordinates before aggregation gives:

```math
\left\{
h_{ij}
\right\}_{j=1}^{n_i}.
```

There are two distinct information losses:

```math
x_{ij}
\longrightarrow
h_{ij}
```

can remove local morphology, while:

```math
\left(
h_{ij},c_{ij}
\right)
\longrightarrow
h_{ij}
```

removes spatial arrangement.

## Three Collision Locations

### Representation Collision

```math
f_{\widehat\theta}(x)
=
f_{\widehat\theta}(x')
```

for target-distinct patches.

### Aggregation Collision

```math
\widehat\nu_a
\ne
\widehat\nu_b,
\qquad
T_{\psi}
\left(
\widehat\nu_a
\right)
=
T_{\psi}
\left(
\widehat\nu_b
\right).
```

### Supervision Collision

Different latent slide states share the same observed weak label:

```math
U_a
\ne
U_b,
\qquad
Y_a
=
Y_b.
```

These failures require different repairs.

## Frozen Encoder Gradient Barrier

For slide loss:

```math
\mathcal{L}_{\mathrm{slide}}
\left(
\psi;
\widehat\theta
\right),
```

a frozen encoder enforces:

```math
\nabla_{\theta}
\mathcal{L}_{\mathrm{slide}}
=
0.
```

Only:

```math
\nabla_{\psi}
\mathcal{L}_{\mathrm{slide}}
```

is used. The downstream task can reweight features but cannot reshape their
local geometry.

## Fine-Tuning Restores A Gradient Path

With joint optimization:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{slide}}
\left(
\psi,\theta
\right)
+
\lambda
\mathcal{L}_{\mathrm{pretrain}}
\left(
\theta
\right),
```

the encoder receives:

```math
\nabla_{\theta}
\mathcal{L}
=
\nabla_{\theta}
\mathcal{L}_{\mathrm{slide}}
+
\lambda
\nabla_{\theta}
\mathcal{L}_{\mathrm{pretrain}}.
```

Now the slide objective can separate patch states that matter to the task, but
large bags make full end-to-end optimization expensive.

## Rare-Instance Attenuation

Suppose a positive slide contains `r_i` rare target patches among
`n_i` total patches:

```math
\pi_i
=
\frac{r_i}{n_i}.
```

If encoder noise maps rare embeddings into the common-patch feature
distribution:

```math
p_H
\left(
h
\mid
U=\mathrm{rare}
\right)
\approx
p_H
\left(
h
\mid
U=\mathrm{common}
\right),
```

then even an ideal aggregator has little evidence. The relevant discriminability
is measured after pushforward, for example:

```math
D_H^{(\mathrm{rare},\mathrm{common})}
=
D
\left(
p_H
\left(
\cdot
\mid
\mathrm{rare}
\right)
\middle\|
p_H
\left(
\cdot
\mid
\mathrm{common}
\right)
\right),
```

not in raw pixel space.

## Domain Shift At The Interface

For source and target image distributions:

```math
p_X^{(s)},
\qquad
p_X^{(t)},
```

the slide learner sees:

```math
p_H^{(s)}
=
\left(
f_{\widehat\theta}
\right)_{\#}
p_X^{(s)},
```

```math
p_H^{(t)}
=
\left(
f_{\widehat\theta}
\right)_{\#}
p_X^{(t)}.
```

Successful augmentation may reduce nuisance shift in feature space, but this
requires a feature-space divergence and a defined baseline:

```math
D_H^{(s,t)}
=
D
\left(
p_H^{(s)},
p_H^{(t)}
\right)
```

The expression is not directly comparable with a raw-image divergence because
the two distributions live in different spaces. One can compare two
feature-space encoders, or evaluate a specified domain-discrimination or
calibration metric. Shortcut-sensitive contrastive geometry can instead
preserve or amplify the feature-space shift.

## DSMIL Interface

DSMIL uses:

```math
f_{\mathrm{SimCLR}}
\longrightarrow
\text{single- or pyramidal multiscale features}
\longrightarrow
\text{dual-stream MIL}.
```

The contrastive encoder and MIL aggregator are separately trained operators.

## CTransPath Interface

For WSI classification, the CTransPath paper uses:

```math
f_{\mathrm{SRCL\mbox{-}CTransPath}}
\longrightarrow
\text{patch features}
\longrightarrow
\text{CLAM-SB attention aggregation}.
```

The paper freezes or transfers the patch feature extractor depending on the
downstream task. Its weakly supervised WSI experiment separates feature
extraction from attention-based slide aggregation.

## Stacked C/R/G/S

```text
patch \mathcal{G}:
    augmentation and candidate or mined-positive relations

patch \mathcal{C}:
    ResNet or CNN-Swin encoder

patch \mathcal{R}:
    retained patch embedding

patch \mathcal{S}:
    SimCLR or SRCL pretraining objective

slide \mathcal{G}:
    bag membership and any retained coordinates or scale relations

slide \mathcal{C}:
    DSMIL or CLAM context and weighting

slide \mathcal{R}:
    bag embedding or bag score

slide \mathcal{S}:
    slide-level label
```

## Debugging Rule

When a WSI model fails, test the interface in order:

1. can the frozen patch features separate the target morphologies?
2. does the bag operator preserve the rare or distributed evidence?
3. does the weak label identify the desired latent state?

Replacing the aggregator cannot repair a failure proven at the first step.
