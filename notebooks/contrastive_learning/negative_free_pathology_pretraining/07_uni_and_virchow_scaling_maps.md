# UNI And Virchow Scaling Maps

Primary anchors:

- Chen et al. "Towards a General-Purpose Foundation Model for Computational
  Pathology." Nature Medicine 2024.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/
- Vorontsov et al. "Virchow: A Million-Slide Digital Pathology Foundation
  Model." 2023. https://arxiv.org/abs/2309.07778

These models instantiate DINOv2 at pathology scale. Their main mathematical
change is not a new loss equation; it is the joint change in model class,
sampling distribution, prototype dimension, and downstream readout.

## UNI Map

UNI pretrains a ViT-L on Mass-100K:

```math
100{,}426\ \mathrm{WSIs}
\longrightarrow
>100\ \mathrm{million\ tissue\ patches}
\longrightarrow
\mathrm{ViT\text{-}L}.
```

The source spans 20 major tissue types. Its training objective combines the
DINOv2 image-level self-distillation and masked image modeling components:

```math
\mathcal{L}_{\mathrm{UNI}}
=
\mathcal{L}_{\mathrm{DINOv2}}
\quad
\text{on}
\quad
p_{\mathrm{Mass\text{-}100K}}(x).
```

## Virchow Map

Virchow uses:

```math
1.5\ \mathrm{million\ WSIs}
\longrightarrow
\mathrm{ViT\text{-}H/14}
\quad
\text{with}
\quad
632\ \mathrm{million\ parameters}.
```

Its DINOv2 projection heads use:

```math
K
=
131{,}072
```

prototype coordinates. During distributed training, each GPU samples one WSI
and 256 foreground tiles from that WSI.

## Virchow Batch Hierarchy

Let GPU index be `g` and sampled slide be `S_g`. Then:

```math
X_{g,1},\ldots,X_{g,256}
\sim
p(x\mid S_g).
```

Across GPUs:

```math
S_g
\sim
p_{\mathrm{slide}}.
```

Tiles are conditionally clustered rather than iid from the marginal tile
distribution. Batch centering and KoLeo spacing therefore combine within-slide
and across-slide statistics.

## Virchow Exported Embedding

For 224-pixel input, Virchow concatenates the class token with the mean of 256
patch tokens:

```math
e(x)
=
\left[
h_{\mathrm{CLS}}(x);
\frac{1}{256}
\sum_{j=1}^{256}
h_j^{\mathrm{patch}}(x)
\right]
\in
\mathbb{R}^{2560}.
```

This readout preserves both a learned global token and the first moment of
local tokens. It is not simply the DINOv2 class token used during global
distillation.

## Scale Does Not Change Identifiability

Larger data and models can reduce approximation and estimation error under the
pretraining distribution. They do not alter the invariances imposed by crops,
masks, teacher targets, or tile sampling:

```math
\text{more scale}
\not\Rightarrow
\text{recovery of information erased by the view kernel}.
```

## Slide Interface

Both models export patch encoders. For slide prediction:

```math
z_i
=
\mathcal{R}_{\mathrm{slide}}
\left(
\left\{
e(x_{ij})
\right\}_{j=1}^{n_i}
\right).
```

The slide inductive bias belongs to `R_slide`, not to DINOv2 pretraining alone.
