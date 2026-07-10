# PLIP And The OpenPath Pair Distribution

Primary anchor:

- Huang et al. "A Visual-Language Foundation Model for Pathology Image
  Analysis Using Medical Twitter." Nature Medicine 2023.
  https://www.nature.com/articles/s41591-023-02504-3

## The Objective Is Familiar; The Sampling Law Is New

PLIP fine-tunes a pretrained CLIP model with the symmetric contrastive
objective. Its pathology-specific mathematical object is therefore primarily
the observed pair distribution:

```math
p_{\mathrm{OpenPath}}(x,t)
=
\sum_{s\in\mathcal{S}}
\pi_s
p_s(x,t),
```

where the source variable includes pathology tweets, selected replies, and
PathLAION retrievals.

The reported OpenPath corpus contains 208,414 image-text pairs:

```math
N_{\mathrm{tweet}}=116{,}504,
```

```math
N_{\mathrm{reply}}=59{,}869,
```

```math
N_{\mathrm{PathLAION}}=32{,}041.
```

## Source-Conditional Semantics

Tweet pairs approximate:

```math
p_{\mathrm{tweet}}(x,t)
=
p
\left(
x,t
\mid
\text{pathology hashtag and collection filters}
\right).
```

Reply pairs add a social selection event. The chosen reply is constrained by
availability, ICD-related keyword filtering, and engagement:

```math
p_{\mathrm{reply}}(x,t)
=
p
\left(
x,t
\mid
A_{\mathrm{reply}}=1
\right).
```

PathLAION pairs are retrieved through a baseline CLIP image geometry. If
`r_{\omega_0}` denotes that retrieval rule:

```math
p_{\mathrm{PathLAION}}(x,t)
=
p
\left(
x,t
\mid
r_{\omega_0}(x;\mathcal{Q})=1
\right),
```

where `\mathcal{Q}` is the pathology query-image set.

## Missingness Is Not Random

Let `A` indicate inclusion after collection and cleaning. The learned model
sees:

```math
p(x,t\mid A=1)
=
\frac{p(A=1\mid x,t)p(x,t)}{p(A=1)}.
```

Hashtags, availability, text cleaning, deduplication, question filtering, and
pathology relevance all contribute to `p(A=1\mid x,t)`. Consequently:

```math
p(x,t\mid A=1)
\ne
p_{\mathrm{clinical}}(x,t)
```

unless a strong transport assumption holds.

## A Pair Can Contain Three Information Types

Decompose text conceptually as:

```math
T
=
\left(
T_{\mathrm{visible}},
T_{\mathrm{case}},
T_{\mathrm{social}}
\right).
```

Only `T_visible` must be grounded in the pixels. Case history and social
discussion may still correlate with the image through selection. Contrastive
training can exploit all three because the loss rewards pair identification,
not localized visual entailment.

## Resolution Operator

PLIP uses a ViT-B/32 image tower with 224-by-224 input. Let native pathology
content be `X^{\star}` and preprocessing be `D`:

```math
X
=
D
\left(
X^{\star}
\right).
```

The learned compatibility is actually:

```math
s
\left(
D(X^{\star}),T
\right).
```

Any morphology erased by `D` cannot be restored by a better contrastive loss.

## C/R/G/S Placement

```math
\mathcal{C}
=
\text{CLIP image and text encoders},
```

```math
\mathcal{R}
=
\text{global normalized embeddings},
```

```math
\mathcal{G}
=
\text{cross-modal angular geometry},
```

```math
\mathcal{S}
=
\text{OpenPath source mixture and diagonal pair identity}.
```

PLIP demonstrates that changing `S` can specialize a model even when the loss
family remains CLIP.
