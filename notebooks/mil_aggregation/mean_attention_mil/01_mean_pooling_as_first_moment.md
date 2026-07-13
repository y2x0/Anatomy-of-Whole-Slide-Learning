# Mean Pooling as a First Moment

## 1. Readout

For contextualized instances `r_ij`,

```math
z_i^{\mathrm{mean}}
=\frac1{n_i}\sum_{j=1}^{n_i}r_{ij}.
```

This is the empirical first moment of the instance representation.

## 2. Collision

Two bags with equal means collide:

```math
\frac1n\sum_jr_j
=\frac1m\sum_ku_k
```

even when their patch distributions, rare extremes, and spatial arrangements
differ substantially.

## 3. Context Changes the Object

Mean after graph or transformer context is not mean of raw patches:

```math
z_i=\frac1{n_i}\sum_j\mathcal C_j(H_i;G_i).
```

The context operator can encode interactions before the first-moment bottleneck.

## 4. WSI Failure

If positive evidence occupies fraction `rho_i` of a slide,

```math
z_i
=\rho_i\,\mathbb E[r\mid z=1]
+(1-\rho_i)\,\mathbb E[r\mid z=0].
```

Small `rho_i` produces dilution unless positive and negative states are strongly
separated in the learned representation.

