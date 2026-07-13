# ProtoPNet Similarity Geometry

ProtoPNet converts local latent resemblance into prototype-unit activations.
Let a convolutional encoder produce

```math
z=f(x)\in\mathbb R^{H\times W\times d},
\qquad
\mathcal P(z)=\{\widetilde z_r\in\mathbb R^d\}_{r=1}^{HW}.
```

For prototype vector `p_k` in `d` dimensions, define squared distances and the
paper's monotone similarity transform:

```math
d_{rk}=\|\widetilde z_r-p_k\|_2^2,
\qquad
s_{rk}=\log\left(\frac{d_{rk}+1}{d_{rk}+\epsilon}\right),
\qquad 0<\epsilon<1.
```

Because

```math
\frac{\partial s}{\partial d}
=\frac{1}{d+1}-\frac{1}{d+\epsilon}
=\frac{\epsilon-1}{(d+1)(d+\epsilon)}<0,
```

similarity ranking is exactly inverse distance ranking. The prototype unit is

```math
g_k(z)=\max_r s_{rk}
=s\left(\min_r d_{rk}\right).
```

## Surviving Statistic

Only the nearest latent patch survives for each prototype. If

```math
r_k^\star\in\arg\min_r d_{rk},
```

then all nonmaximal local matches have zero direct effect on `g_k`, away from
ties. The explanation is therefore existential:

```math
\exists r:\widetilde z_r\approx p_k,
```

not distributional. Ten moderate matches and one strong match can produce the
same prototype activation.

## Local Geometry

For a unique winning patch,

```math
\nabla_{\widetilde z_{r_k^\star}}g_k
=2(\widetilde z_{r_k^\star}-p_k)
\left[
\frac{1}{d_{r_k^\star k}+1}
-\frac{1}{d_{r_k^\star k}+\epsilon}
\right].
```

The gradient vanishes for every losing location. At a tie the max is
nondifferentiable and the displayed patch can switch under an arbitrarily small
perturbation. A prototype heatmap is consequently a metric map followed by a
winner-take-all readout, not a complete account of image evidence.

## WSI Translation

Applied to patch embeddings, the same map becomes

```math
g_{ik}=\max_{1\le j\le n_i}
\log\left(
\frac{\|h_{ij}-p_k\|_2^2+1}
{\|h_{ij}-p_k\|_2^2+\epsilon}
\right).
```

Its sparse-positive sensitivity is useful, but it also makes the explanation
fragile to artifacts that become accidental nearest neighbors.
