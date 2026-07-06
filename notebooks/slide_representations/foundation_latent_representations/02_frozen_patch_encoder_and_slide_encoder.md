# Frozen Patch Encoder And Slide Encoder

Many pathology pipelines use foundation models in two different ways.

## Frozen Patch Encoder

First tile the slide:

```math
S_i
\mapsto
\{x_{ij},c_{ij}\}_{j=1}^{n_i}.
```

Then encode patches:

```math
h_{ij}
=
E_{\operatorname{FM}}(x_{ij}).
```

The slide still needs an aggregator:

```math
z_i
=
\mathcal{R}_\theta(\{h_{ij},c_{ij}\}_{j=1}^{n_i}).
```

Prediction:

```math
\widehat y_i
=
\mathcal{H}_\theta(z_i).
```

This is not yet a slide foundation representation. It is a standard WSI model
using foundation patch features.

## Slide Encoder

A slide encoder maps the set or sequence of patch tokens into a pretrained
slide-level object:

```math
z_i
=
F_{\operatorname{slide}}
\left(
\{h_{ij},c_{ij}\}_{j=1}^{n_i}
\right).
```

If $F_{\operatorname{slide}}$ is pretrained, the slide representation itself is
foundation-level:

```math
z_i\in\mathcal{Z}_{\operatorname{slide}}.
```

GigaPath-style models fit here: a tile encoder produces patch features, and a
slide encoder produces slide-level embeddings from tile features and positions.

## Hierarchical Slide Encoder

Some models introduce intermediate region tokens:

```math
u_{ir}
=
\mathcal{R}_{\operatorname{region}}
\left(
\{h_{ij}:j\in R_r\}
\right).
```

Then:

```math
z_i
=
\mathcal{R}_{\operatorname{slide}}
\left(
\{u_{ir}\}_{r=1}^{R_i}
\right).
```

HIPT-style models are naturally hierarchical:

```text
small visual tokens -> patch tokens -> region tokens -> slide task representation
```

The representation object is multiscale, not just a bag of patch embeddings.

## Frozen Encoder With Trainable Aggregator

The common practical pipeline is:

```math
E_{\operatorname{FM}}
\quad
\text{frozen},
\qquad
\mathcal{R}_\theta,\mathcal{H}_\theta
\quad
\text{trained}.
```

Gradient flow:

```math
\nabla E_{\operatorname{FM}}=0,
\qquad
\nabla\mathcal{R}_\theta\ne0.
```

This preserves the pretrained patch geometry but learns a task-specific slide
geometry.

## Fine-Tuned Slide Geometry

If the encoder is updated:

```math
\nabla E_{\operatorname{FM}}\ne0,
```

then patch geometry changes:

```math
d_{\operatorname{FM}}(x_a,x_b)
\to
d_{\theta}(x_a,x_b).
```

Fine-tuning can improve task fit but may destroy transferability or collapse
rare morphology structure.

## Dense Summary

```text
frozen patch FM:
    pretrained patch space plus new slide aggregator

pretrained slide FM:
    pretrained patch and slide space

adapted FM:
    pretrained initialization with task-deformed geometry
```

The representation question is:

```text
Which part of the slide map was learned before the downstream task?
```
