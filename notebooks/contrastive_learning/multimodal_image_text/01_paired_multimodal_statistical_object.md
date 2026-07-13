# The Paired Multimodal Statistical Object

Image-text contrast begins with an observed pair, but the observed pair is not
necessarily a statement that every visual feature is described by every word.
That distinction is central in pathology, where captions may describe a case,
a figure panel, a differential diagnosis, or clinical context outside the
crop.

Primary anchor:

- Radford et al. "Learning Transferable Visual Models From Natural Language
  Supervision." ICML 2021. https://arxiv.org/abs/2103.00020

## Observed Pair And Latent Compatibility

Let an image and text be random variables:

```math
X
\in
\mathcal{X},
\qquad
T
\in
\mathcal{T}.
```

The training corpus supplies samples:

```math
\mathcal{D}_N
=
\left\{
\left(x_i,t_i\right)
\right\}_{i=1}^{N}
\sim
p_{\mathrm{obs}}(x,t).
```

Introduce a latent semantic state `Y`, a visual nuisance `U`, and a textual
nuisance `V`:

```math
Y\sim p(y),
\qquad
X\sim p(x\mid Y,U),
\qquad
T\sim p(t\mid Y,V).
```

A clean paired model would imply:

```math
p_{\mathrm{clean}}(x,t)
=
\int
p(y)
p(x\mid y)
p(t\mid y)
\,\mathrm{d}y.
```

Pathology corpora instead often contain a mixture:

```math
p_{\mathrm{obs}}(x,t)
=
\left(1-\varepsilon\right)
p_{\mathrm{clean}}(x,t)
+
\varepsilon
p_{\mathrm{noise}}(x,t).
```

The noise term need not be independent pairing. It can be clinically related
but visually ungrounded text, a caption belonging to another subfigure, or a
retrieval-selected pseudo-pair.

## Pair Identity Is A Supervision Relation

For a minibatch of size `B`, define the observed relation matrix:

```math
R_{ij}
=
\mathbb{1}
\left\{
i=j
\right\}.
```

Standard CLIP treats the diagonal as positive and every off-diagonal entry as
negative. A latent semantic relation would instead be:

```math
R^{\star}_{ij}
=
\mathbb{1}
\left\{
Y_i\sim Y_j
\right\},
```

where `\sim` can mean same diagnosis, compatible morphology, or another
task-specific equivalence. In general:

```math
R
\ne
R^{\star}.
```

The discrepancy has two directions:

```math
R_{ij}=1,
\quad
R^{\star}_{ij}=0
\qquad
\text{pair contamination},
```

```math
R_{ij}=0,
\quad
R^{\star}_{ij}=1
\qquad
\text{false negative}.
```

## Encoders And Surviving Object

Let normalized image and text embeddings be:

```math
u_{\theta}(x)
=
\frac{P_x f_{\theta}(x)}
{\left\|P_x f_{\theta}(x)\right\|_2},
\qquad
v_{\phi}(t)
=
\frac{P_t g_{\phi}(t)}
{\left\|P_t g_{\phi}(t)\right\|_2}.
```

The score is:

```math
s_{\theta,\phi}(x,t)
=
\gamma
u_{\theta}(x)^{\top}v_{\phi}(t),
```

with positive logit scale `\gamma`. The surviving object is therefore not a
caption generator or a calibrated disease posterior. It is a cross-modal
angular compatibility function.

## Three Pairing Regimes

One-to-one observed pairing:

```math
\mathcal{P}_i
=
\left\{
t_i
\right\}.
```

One-to-many semantic pairing:

```math
\mathcal{P}^{\star}_i
=
\left\{
t:
t\text{ is compatible with }x_i
\right\},
\qquad
\left|\mathcal{P}^{\star}_i\right|>1.
```

Bag-to-bag pairing:

```math
\mathcal{B}^{x}_i
\longleftrightarrow
\mathcal{B}^{t}_i.
```

CLIP instantiates the first, pathology corpora frequently behave like the
second, and CPLIP explicitly optimizes the third.

## C/R/G/S Placement

```math
\mathcal{C}
=
\left(f_{\theta},g_{\phi}\right),
\qquad
\mathcal{R}
=
\left(P_x,P_t,\ell_2\text{-normalization}\right),
```

```math
\mathcal{G}
=
\left\{
u^{\top}v
\right\},
\qquad
\mathcal{S}
=
R.
```

The most dangerous modeling choice can therefore live in `S`, even when all
architectural attention is directed at `C`.
