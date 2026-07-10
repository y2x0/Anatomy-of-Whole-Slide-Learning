# DSMIL SimCLR Pretraining Map

DSMIL contains two mathematically separate learners:

1. a patch encoder pretrained by SimCLR;
2. a dual-stream MIL aggregator trained from slide labels.

Conflating them obscures where the contrastive inductive bias enters.

Primary anchors:

- DSMIL: https://arxiv.org/abs/2011.08939
- SimCLR: https://arxiv.org/abs/2002.05709

## Paper-Level Data Construction

For a magnification level `m`, DSMIL densely crops non-overlapping WSI
patches and treats every crop as an individual SimCLR image:

```math
\mathcal{D}^{(m)}
=
\left\{
x_n^{(m)}
\right\}_{n=1}^{N_m}.
```

The paper trains feature extractors separately across magnifications. Slide
labels are not used in this pretraining stage.

For each source patch:

```math
\widetilde x_{n,1}^{(m)}
\sim
\mathcal{A}
\left(
\cdot
\mid
x_n^{(m)}
\right),
\qquad
\widetilde x_{n,2}^{(m)}
\sim
\mathcal{A}
\left(
\cdot
\mid
x_n^{(m)}
\right).
```

## SimCLR Representation Map

Let the backbone and projection head be:

```math
h_{n,a}^{(m)}
=
f_{\theta_m}
\left(
\widetilde x_{n,a}^{(m)}
\right),
```

```math
z_{n,a}^{(m)}
=
g_{\phi_m}
\left(
h_{n,a}^{(m)}
\right),
\qquad
u_{n,a}^{(m)}
=
\frac{
z_{n,a}^{(m)}
}{
\left\lVert
z_{n,a}^{(m)}
\right\rVert_2
}.
```

For a minibatch of `B` patch identities, define:

```math
s
\left(
u,v
\right)
=
u^{\top}v.
```

The directed NT-Xent loss from view `(n,a)` to its paired view
`(n,b)` is:

```math
\ell_{n,a\rightarrow b}
=
-
\log
\frac{
\exp
\left(
u_{n,a}^{\top}u_{n,b}/\tau
\right)
}{
\sum\limits_{
\substack{
(r,c)\\
(r,c)\ne(n,a)
}
}
\exp
\left(
u_{n,a}^{\top}u_{r,c}/\tau
\right)
}.
```

The symmetric batch objective is:

```math
\mathcal{L}_{\mathrm{SimCLR}}
=
\frac{1}{2B}
\sum_{n=1}^{B}
\left(
\ell_{n,1\rightarrow 2}
+
\ell_{n,2\rightarrow 1}
\right).
```

DSMIL invokes this SimCLR construction rather than introducing a distinct
contrastive objective.

## What Is Retained

After contrastive training, DSMIL retains the feature extractor:

```math
f_{\theta_m}
:
\mathcal{X}^{(m)}
\longrightarrow
\mathbb{R}^{d_m},
```

and computes patch embeddings:

```math
h_{ij}^{(m)}
=
f_{\theta_m}
\left(
x_{ij}^{(m)}
\right).
```

The projection head used to define contrastive angles is not the slide
aggregator. The paper-level downstream object is the backbone feature.

## Scale-Specific Pretraining

For two magnifications, such as high `h` and low `\ell`, the
pretraining objectives factor:

```math
\mathcal{L}_{\mathrm{pretrain}}
=
\mathcal{L}_{\mathrm{SimCLR}}^{(h)}
+
\mathcal{L}_{\mathrm{SimCLR}}^{(\ell)},
```

with distinct image collections and feature extractors.

No contrastive term in the DSMIL paper directly aligns:

```math
h_{ij}^{(h)}
\quad
\text{with}
\quad
h_{ik}^{(\ell)}.
```

Cross-scale correspondence enters later through pyramidal feature
construction.

## Pyramidal Feature Construction

Let `\pi(j)` map a high-magnification patch to the lower-magnification
patch containing it. DSMIL duplicates the low-magnification feature for the
corresponding high-resolution subpatches and concatenates:

```math
v_{ij}
=
\left[
h_{ij}^{(h)}
;
h_{i,\pi(j)}^{(\ell)}
\right].
```

The slide bag becomes:

```math
\mathcal{V}_i
=
\left\{
v_{ij}
\right\}_{j=1}^{n_i^{(h)}}.
```

The local cross-scale constraint is therefore carried by the deterministic
parent map `\pi`, not by the SimCLR loss.

## Downstream MIL Stage

DSMIL then applies its dual-stream aggregator:

```math
\widehat y_i
=
\mathcal{M}_{\psi}
\left(
\mathcal{V}_i
\right).
```

The slide loss is:

```math
\mathcal{L}_{\mathrm{slide}}
=
\sum_i
\ell_{\mathrm{sup}}
\left(
\widehat y_i,y_i
\right).
```

The complete composition is:

```math
\left\{
x_{ij}^{(m)}
\right\}
\xrightarrow{
\text{unlabeled SimCLR}
}
\left\{
h_{ij}^{(m)}
\right\}
\xrightarrow{
\text{parent concatenation}
}
\mathcal{V}_i
\xrightarrow{
\text{DSMIL}
}
\widehat y_i.
```

## Optimization Separation

The two stages optimize different empirical risks:

```math
\widehat\theta
\in
\arg\min_{\theta}
\widehat{\mathcal{L}}_{\mathrm{SimCLR}}
\left(
\theta
\right),
```

followed by:

```math
\widehat\psi
\in
\arg\min_{\psi}
\widehat{\mathcal{L}}_{\mathrm{slide}}
\left(
\psi;
\widehat\theta
\right).
```

If the encoder is frozen during MIL training, the slide objective cannot repair
a discarded patch distinction:

```math
f_{\widehat\theta}(x)
=
f_{\widehat\theta}(x')
\quad
\Longrightarrow
\quad
\mathcal{M}_{\psi}
\text{ receives identical instance features for }x\text{ and }x'.
```

## Paper-Specific Implementation Facts

The DSMIL paper reports:

```text
patch size:
    224 by 224

patch overlap:
    none in the stated WSI preprocessing

SimCLR backbone:
    ResNet-18

SimCLR minibatch:
    512

MIL minibatch:
    one bag
```

These are implementation choices, not consequences of the objective.

## Why Contrastive Pretraining Helps Large Bags

End-to-end patch encoding for a bag of size `n_i` requires storing
activation tensors for many instances:

```math
\mathrm{memory}_{\mathrm{end\mbox{-}to\mbox{-}end}}
\propto
n_i
\cdot
\mathrm{activation\ size}.
```

Precomputing frozen features changes the slide-stage memory to:

```math
\mathrm{memory}_{\mathrm{MIL}}
\propto
n_i d.
```

This is a computational benefit of stage separation. It is not evidence that
the patch objective is slide-optimal.

## C/R/G/S At Both Stages

```text
pretraining \mathcal{G}:
    two augmented views of each patch and in-batch patch candidates

pretraining \mathcal{C}:
    ResNet-18 patch encoder

pretraining \mathcal{R}:
    SimCLR projector and unit-normalized embedding

pretraining \mathcal{S}:
    symmetric same-patch NT-Xent

slide \mathcal{G}:
    a bag of precomputed single- or multiscale patch features

slide \mathcal{C}:
    DSMIL critical-instance and relation streams

slide \mathcal{R}:
    dual-stream bag scores

slide \mathcal{S}:
    slide label
```

The method is contrastively pretrained and weakly supervised, but those
supervision operators act on different objects.
