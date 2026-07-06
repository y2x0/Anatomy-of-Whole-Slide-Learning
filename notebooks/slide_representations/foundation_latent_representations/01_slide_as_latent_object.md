# Slide As Latent Object

A foundation encoder defines a representation map before the downstream task.

Patch-level:

```math
h_{ij}
=
E_{\text{FM}}(x_{ij}).
```

Slide-level:

```math
z_i
=
F_{\text{FM}}(\{x_{ij},c_{ij}\}_{j=1}^{n_i}).
```

The slide is then represented as:

```math
\mathcal{X}_i=z_i\in\mathbb{R}^{D}.
```

or, if patch tokens are retained:

```math
\mathcal{X}_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}\in\mathcal{Z}_{\text{FM}}.
```

## Pretrained Geometry

The latent space has a geometry induced by pretraining:

```math
d_{\text{FM}}(a,b)
=
\|E_{\text{FM}}(a)-E_{\text{FM}}(b)\|.
```

This distance is not neutral. It encodes invariances and sensitivities learned
from the pretraining objective and data.

If pretraining uses contrastive learning:

```math
\ell
=
-\log
\frac{\exp(\mathrm{sim}(z_i,z_i^+)/\tau)}
{\sum_k\exp(\mathrm{sim}(z_i,z_k^-)/\tau)}.
```

then the space is shaped by positive and negative pair definitions.

If pretraining uses masked modeling:

```math
\ell
=
\sum_{j\in M}
\|x_{ij}-\widehat x_{ij}\|^2
```

or token prediction, then the space is shaped by reconstruction or prediction of
masked content.

If pretraining uses image-text alignment:

```math
\ell
=
-\log
\frac{\exp(z_i^\top t_i/\tau)}
{\sum_k\exp(z_i^\top t_k/\tau)}.
```

then image geometry is partly text geometry.

## Frozen Versus Adapted

A frozen representation uses:

```math
z_i=F_{\text{FM}}(S_i),
\qquad
\widehat y_i=\mathcal{H}_\theta(z_i),
```

where only $\mathcal{H}_\theta$ is trained.

Adaptation changes the representation:

```math
z_i
=
F_{\mathrm{FM},\theta}(S_i).
```

The mathematical object differs:

```text
frozen:
    slide in fixed pretrained geometry

adapted:
    slide in task-deformed geometry
```

## Latent Object Types

Patch-token latent:

```math
\mathcal{X}_i=\{h_{ij}\}_{j=1}^{n_i}.
```

Slide-vector latent:

```math
\mathcal{X}_i=z_i.
```

Slide-token latent:

```math
\mathcal{X}_i=\{u_{ir}\}_{r=1}^{R}.
```

Multimodal latent:

```math
\mathcal{X}_i=(z_i,\mathcal{T}),
```

where $\mathcal{T}$ is a text embedding space aligned to image space.

Retrieval latent:

```math
\mathcal{X}_i=(z_i,\mathcal{M}(z_i)).
```

## Dense Summary

```math
\boxed{
\mathcal{X}_i
=
F_{\text{FM}}(S_i)
}
```

A slide-as-latent-object model inherits the pretraining model's geometry before
the downstream task sees any labels.
