# PANTHER EM, Slide Representation, and Maps

## 1. E-Step

Given parameters at iteration `t`, the posterior responsibility is

```math
q_{ijk}^{(t+1)}
=p(z_{ij}=k\mid h_{ij};\theta_i^{(t)})
=\frac{
\pi_{ik}^{(t)}
\mathcal N(h_{ij};\mu_{ik}^{(t)},\Sigma_{ik}^{(t)})
}{
\sum_{\ell=1}^K\pi_{i\ell}^{(t)}
\mathcal N(h_{ij};\mu_{i\ell}^{(t)},\Sigma_{i\ell}^{(t)})
}.
```

Responsibilities are normalized across components for one patch. They answer
which component best explains the embedding under the mixture, not which patch
caused the slide prediction.

## 2. M-Step

Let

```math
N_{ik}=\sum_{j=1}^{n_i}q_{ijk}.
```

Then

```math
\widehat\pi_{ik}=\frac{N_{ik}}{n_i},
\qquad
\widehat\mu_{ik}
=\frac{1}{N_{ik}}\sum_jq_{ijk}h_{ij},
```

```math
\widehat\Sigma_{ik}
=\mathrm{diag}\left(
\frac{1}{N_{ik}}
\sum_jq_{ijk}(h_{ij}-\widehat\mu_{ik})^{\odot2}
\right).
```

The mixture weight is a soft empirical prevalence. The mean and variance are
responsibility-weighted moments. When the effective component count `N_ik` is
small, both are high-variance
estimates even if their component index appears interpretable.

## 3. Spatial Visualization

A hard prototype map uses

```math
\widehat z_{ij}=\arg\max_k q_{ijk}.
```

This map visualizes mixture assignment after embeddings have been computed.
The GMM itself has no spatial context: permuting patch coordinates leaves every
estimated parameter unchanged. Contiguous colors can reflect spatially
coherent morphology, but contiguity was not inferred by the mixture.

## 4. Downstream Credit

For prediction

```math
F_{ic}=H_c(z_i),
```

neither responsibility `q_ijk` nor estimated prevalence `pi_ik` is
automatically class credit. If
the head is block-additive,

```math
F_{ic}=b_c+\sum_k h_{ck}(z_{ik}),
```

then the `k`th block term is exact prototype-block score credit. Under a general
MLP, component interactions destroy that direct decomposition.

## 5. Counterexample

Two slides may have equal estimated component prevalence but opposite
predictions because
their component means differ. Conversely, a rare component can dominate a
nonlinear head. Therefore prevalence maps must not be labeled importance maps
without head-specific analysis.
