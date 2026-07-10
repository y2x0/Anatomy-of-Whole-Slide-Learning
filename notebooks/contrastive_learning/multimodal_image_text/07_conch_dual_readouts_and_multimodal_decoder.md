# CONCH Dual Readouts And Multimodal Decoder

Primary anchor:

- Lu et al. "A Visual-Language Foundation Model for Computational Pathology."
  Nature Medicine 2024. https://arxiv.org/abs/2307.12914

## Separate Unimodal Initialization

CONCH initializes its image and language components through separate
pretraining before joint image-caption learning. Abstractly:

```math
\theta_0
\leftarrow
\argmin_{\theta}
\mathcal{L}_{\mathrm{iBOT}}(\theta),
```

```math
\phi_0
\leftarrow
\argmin_{\phi}
\mathcal{L}_{\mathrm{LM}}(\phi).
```

Joint training then starts from modality-specific representations rather than
randomly initialized towers.

## Image Token Tensor

For image `x_i`, a ViT-B image encoder returns patch tokens:

```math
H_i
=
f_{\theta}(x_i)
\in
\mathbb{R}^{L_x\times d_x}.
```

CONCH applies two distinct attentional poolers to the same image-token tensor.

The contrastive pooler has one learned query:

```math
q_{\mathrm{ctr}}
\in
\mathbb{R}^{1\times d_x},
```

and returns one global image token:

```math
h_i^{\mathrm{ctr}}
=
\mathrm{AttnPool}_{\mathrm{ctr}}
\left(
q_{\mathrm{ctr}},H_i
\right)
\in
\mathbb{R}^{d_x}.
```

The caption pooler uses 256 learned queries:

```math
Q_{\mathrm{cap}}
\in
\mathbb{R}^{256\times d_x},
```

and returns local visual tokens:

```math
Z_i^{\mathrm{cap}}
=
\mathrm{AttnPool}_{\mathrm{cap}}
\left(
Q_{\mathrm{cap}},H_i
\right)
\in
\mathbb{R}^{256\times d_m}.
```

## Text Global Token

For caption tokens `w_{i,0:T}`, the text encoder produces a global token:

```math
h_i^{\mathrm{text}}
=
g_{\phi}
\left(
w_{i,0:T}
\right)
\in
\mathbb{R}^{d_t}.
```

Projection and normalization give the contrastive pair:

```math
u_i
=
\frac{P_x h_i^{\mathrm{ctr}}}
{\|P_x h_i^{\mathrm{ctr}}\|_2},
\qquad
v_i
=
\frac{P_t h_i^{\mathrm{text}}}
{\|P_t h_i^{\mathrm{text}}\|_2}.
```

## Multimodal Decoder

The decoder is autoregressive. At token position `r`, self-attention summarizes
the text prefix and cross-attention reads the 256 image tokens:

```math
a_{i,r}
=
\mathrm{SelfAttn}
\left(
w_{i,0:r-1}
\right),
```

```math
b_{i,r}
=
\mathrm{CrossAttn}
\left(
a_{i,r},
Z_i^{\mathrm{cap}}
\right),
```

```math
p_{\psi}
\left(
w_{i,r}
\mid
w_{i,0:r-1},x_i
\right)
=
\mathrm{softmax}
\left(
W_o b_{i,r}
\right).
```

## Why Two Image Readouts Matter

The global contrastive path compresses:

```math
H_i
\longmapsto
h_i^{\mathrm{ctr}}
\in
\mathbb{R}^{d_x}.
```

The caption path preserves a larger bottleneck:

```math
H_i
\longmapsto
Z_i^{\mathrm{cap}}
\in
\mathbb{R}^{256\times d_m}.
```

If two images share the same global token but differ in caption tokens:

```math
h_a^{\mathrm{ctr}}
=
h_b^{\mathrm{ctr}},
\qquad
Z_a^{\mathrm{cap}}
\ne
Z_b^{\mathrm{cap}},
```

the contrastive head cannot distinguish them, but the captioning decoder can.
Thus CONCH does not force all visual-language information through one vector.

## Information Boundary

Both losses update the shared backbone:

```math
\nabla_{\theta}\mathcal{L}
=
\nabla_{\theta}\mathcal{L}_{\mathrm{ctr}}
+
\nabla_{\theta}\mathcal{L}_{\mathrm{cap}}.
```

After the branch point, however:

```math
\nabla_{\omega_{\mathrm{ctr}}}
\mathcal{L}_{\mathrm{cap}}
=
0,
\qquad
\nabla_{\omega_{\mathrm{cap}}}
\mathcal{L}_{\mathrm{ctr}}
=
0.
```

The two readouts impose different sufficient-statistic pressures on shared
image tokens.
