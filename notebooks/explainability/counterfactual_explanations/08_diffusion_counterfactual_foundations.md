# Diffusion Counterfactual Foundations

## 1. Conditional Diffusion Autoencoder

Let an encoder map image patch `x_0` to semantic feature `z`, while a forward
diffusion process maps the image toward noise:

```math
q(x_t\mid x_0)
=\mathcal N
(x_t;\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I).
```

A conditional denoiser learns

```math
\epsilon_{\theta}(x_t,t,z)
\approx\epsilon.
```

The original patch can be reconstructed by reversing from its encoded noise
while conditioning on its own feature vector.

## 2. Semantic Edit

A counterfactual changes the condition but retains the image-specific noise:

```math
z_{\mathrm{cf}}=z+\alpha d,
```

```math
x_{\mathrm{cf}}
=\mathrm{DDIM}_{\theta}
(x_T(x_0),z_{\mathrm{cf}}).
```

Holding `x_T` fixed is intended to preserve identity and details unrelated to
the semantic manipulation. Preservation is empirical, not guaranteed by the
forward equation.

## 3. Three Distances

Diffusion counterfactuals have distinct costs:

```math
d_z(z,z_{\mathrm{cf}}),
\qquad
d_x(x_0,x_{\mathrm{cf}}),
\qquad
d_m(m(x_0),m(x_{\mathrm{cf}})),
```

where `m` extracts morphology. A small latent shift can yield a large image
change, and high image similarity can conceal a diagnostically important
cellular edit.

## 4. Generator-Relative Plausibility

The output lies in the range of a trained decoder, but

```math
x_{\mathrm{cf}}\in\mathrm{range}(G_{\theta})
```

does not imply that it is a realizable tissue specimen. Diffusion models can
hallucinate cells, alter counts, or encode cohort-specific stain statistics.

## 5. Minimality

If `alpha_star` is the first amplitude crossing a target threshold on a fixed
direction,

```math
\alpha^{\star}
=\inf\{\alpha\ge0:
f(G(z+\alpha d))\in\mathcal Y_{\star}\},
```

it is minimal only along direction `d`, not globally over the generator
manifold.

