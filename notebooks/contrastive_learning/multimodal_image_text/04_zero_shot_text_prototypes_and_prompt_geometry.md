# Zero-Shot Text Prototypes And Prompt Geometry

Primary anchors:

- Radford et al. "Learning Transferable Visual Models From Natural Language
  Supervision." ICML 2021. https://arxiv.org/abs/2103.00020
- Lu et al. "A Visual-Language Foundation Model for Computational Pathology."
  Nature Medicine 2024. https://arxiv.org/abs/2307.12914

## Text Embeddings Become Classifier Weights

For class `c`, let `q_{c,r}` denote prompt template `r`. Encode and normalize:

```math
v_{c,r}
=
\frac{g_{\phi}(q_{c,r})}
{\left\|g_{\phi}(q_{c,r})\right\|_2}.
```

For a single prompt per class, zero-shot prediction is:

```math
p(c\mid x)
=
\frac{
\exp
\left(
\gamma u(x)^{\top}v_c
\right)
}{
\sum_{k=1}^{C}
\exp
\left(
\gamma u(x)^{\top}v_k
\right)
}.
```

This is a normalized linear classifier with:

```math
w_c
=
\gamma v_c,
\qquad
b_c
=
0.
```

Language does not merely annotate the class. It constructs the classifier
weight vector.

## Decision Hyperplanes

Classes `a` and `b` tie when:

```math
u(x)^{\top}
\left(
v_a-v_b
\right)
=
0.
```

Changing wording changes `v_a-v_b`, hence rotates the decision boundary
without updating the image encoder.

For perturbations `\Delta v_a` and `\Delta v_b`, the margin changes by:

```math
\Delta m_{ab}(x)
=
\gamma
u(x)^{\top}
\left(
\Delta v_a-
\Delta v_b
\right).
```

Prompt sensitivity is therefore classifier sensitivity, not cosmetic language
variation.

## Prompt Ensembling

One common construction averages prompt embeddings and renormalizes:

```math
\bar v_c
=
\frac{1}{R_c}
\sum_{r=1}^{R_c}
v_{c,r},
\qquad
\widetilde v_c
=
\frac{\bar v_c}{\|\bar v_c\|_2}.
```

This differs from averaging class probabilities:

```math
\mathrm{softmax}
\left(
\gamma u^{\top}\widetilde v_c
\right)
\ne
\frac{1}{R_c}
\sum_{r=1}^{R_c}
\mathrm{softmax}
\left(
\gamma u^{\top}v_{c,r}
\right).
```

The first creates one spherical prototype. The second is an ensemble of
classifiers.

## Prototype Cancellation

The norm:

```math
\|\bar v_c\|_2^2
=
\frac{1}{R_c^2}
\sum_{r=1}^{R_c}
\sum_{s=1}^{R_c}
v_{c,r}^{\top}v_{c,s}
```

measures prompt agreement. Contradictory or heterogeneous prompts can make
`\|\bar v_c\|_2` small. Renormalization then magnifies a weak resultant
direction.

## Calibration Is Not Inherited

The zero-shot probability depends on the candidate class set:

```math
p(c\mid x;\mathcal{C})
\ne
p(c\mid x;\mathcal{C}')
```

when `\mathcal{C}` changes. It is a normalized preference among supplied text
prototypes, not a prevalence-calibrated disease probability.

## WSI Consequence

If tile scores are subsequently pooled, the prompt participates twice:

```math
q_{jc}
=
u(x_j)^{\top}v_c,
```

```math
Q_c
=
\mathcal{R}_c
\left(
q_{1c},ldots,q_{nc}
\right).
```

It defines the local evidence direction and, under class-specific top-k
pooling, determines which patches survive aggregation.
