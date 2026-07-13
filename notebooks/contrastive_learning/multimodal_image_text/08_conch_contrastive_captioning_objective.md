# CONCH Contrastive-Captioning Objective

Primary anchor:

- Lu et al. "A Visual-Language Foundation Model for Computational Pathology."
  Nature Medicine 2024. https://arxiv.org/abs/2307.12914

## Exact Joint Loss

For `M` paired images and captions, let normalized global embeddings be `u_i`
and `v_i`. CONCH writes a learned multiplicative logit scale `\tau`:

```math
\tau=e^{\lambda}>0.
```

where `\lambda` is the unconstrained optimization parameter. The logits are:

```math
\ell_{ij}
=
\tau u_i^{\top}v_j.
```

The bidirectional contrastive term is:

```math
\mathcal{L}_{\mathrm{ctr}}
=
-\frac{1}{2M}
\sum_{i=1}^{M}
\log
\frac{
\exp
\left(
\tau u_i^{\top}v_i
\right)
}{
\sum_{j=1}^{M}
\exp
\left(
\tau u_i^{\top}v_j
\right)
}
```

```math
\phantom{\mathcal{L}_{\mathrm{ctr}}=}
-\frac{1}{2M}
\sum_{j=1}^{M}
\log
\frac{
\exp
\left(
\tau v_j^{\top}u_j
\right)
}{
\sum_{i=1}^{M}
\exp
\left(
\tau v_j^{\top}u_i
\right)
}.
```

Let `s_{ij}=u_i^{\top}v_j` and define the image-to-text row distribution:

```math
p_{ij}
=
\frac{\exp(\tau s_{ij})}
{\sum_{k=1}^{M}\exp(\tau s_{ik})}.
```

The row contribution has scale derivative:

```math
\frac{\partial\mathcal{L}_{i\rightarrow t}}
{\partial\tau}
=
\frac{1}{M}
\left(
\sum_{j=1}^{M}p_{ij}s_{ij}-s_{ii}
\right).
```

Because `\tau=e^{\lambda}`, the gradient seen by the unconstrained parameter
is:

```math
\frac{\partial\mathcal{L}_{i\rightarrow t}}
{\partial\lambda}
=
\tau
\frac{\partial\mathcal{L}_{i\rightarrow t}}
{\partial\tau}.
```

The reverse text-to-image term contributes the analogous column expression.
Thus increasing the scale is favored only when the positive similarity is
larger than the row's current probability-weighted similarity; scale learning
is coupled to separability, not an independent post-processing choice.

The autoregressive captioning term is:

```math
\mathcal{L}_{\mathrm{cap}}
=
-\frac{1}{M}
\sum_{i=1}^{M}
\sum_{r=1}^{T_i+1}
\log
p
\left(
w_{i,r}
\mid
w_{i,0:r-1},x_i;
\theta,\phi,\psi
\right).
```

The paper uses an equal-weighted combination:

```math
\mathcal{L}_{\mathrm{CONCH}}
=
\mathcal{L}_{\mathrm{ctr}}
+
\mathcal{L}_{\mathrm{cap}}.
```

Equal scalar coefficients do not imply equal gradient magnitudes:

```math
\left\|
\nabla_{\theta}
\mathcal{L}_{\mathrm{ctr}}
\right\|_2
\ne
\left\|
\nabla_{\theta}
\mathcal{L}_{\mathrm{cap}}
\right\|_2.
```

## Two Conditional Problems

Contrastive learning estimates pair compatibility relative to in-batch
alternatives:

```math
\text{identify }t_i
\text{ among }
\left\{
t_1,\ldots,t_M
\right\}.
```

Captioning estimates next-token conditionals:

```math
p
\left(
w_r
\mid
w_{<r},x
\right).
```

The first can succeed using coarse discriminative cues. The second is
penalized whenever image-conditioned prefix information cannot predict caption
tokens, but it can also learn dataset-specific language regularities.

## Caption Gradient Allocation

At decoder position `r`, let token logits be `a_{i,r,k}` and probabilities be
`\pi_{i,r,k}`. Then:

```math
\frac{\partial\mathcal{L}_{\mathrm{cap}}}
{\partial a_{i,r,k}}
=
\frac{1}{M}
\left(
\pi_{i,r,k}
-
\mathbb{1}
\left\{
k=w_{i,r}
\right\}
\right).
```

Tokens predictable from the text prefix alone can produce weak pressure on
the visual cross-attention path. The relevant quantity is conditional visual
gain:

```math
\Delta_r
=
\log p
\left(
w_r\mid w_{<r},x
\right)
-
\log p
\left(
w_r\mid w_{<r}
\right).
```

Large `\Delta_r` indicates image-grounded predictive value; ordinary caption
likelihood does not explicitly isolate it.

## Pair Construction Is Model-Assisted

CONCH cleans figure-caption data using detection, caption splitting, and a
CLIP matching model. If `A_{ij}` denotes the event that subfigure `i` is
matched to caption fragment `j`, then training conditions on:

```math
A_{ij}
=
\mathbb{1}
\left\{
\mathrm{match}_{\omega_0}(x_i,t_j)
\ge
c
\right\}.
```

Therefore the joint objective is exact relative to the constructed pairs, but
the pair relation itself is an estimated latent alignment.

## Complementary Failure Modes

Contrastive-only failure:

```math
\text{coarse pair discrimination}
\not\Rightarrow
\text{fine visual description}.
```

Captioning-only failure:

```math
\text{language fluency}
\not\Rightarrow
\text{cross-modal retrieval geometry}.
```

Joint training combines pressures; it does not prove that every generated
token is visually grounded or that every embedding direction is clinically
meaningful.
