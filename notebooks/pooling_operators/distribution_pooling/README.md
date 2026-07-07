# Distribution Pooling

Distribution pooling represents a slide by statistics of the full empirical
distribution, not by one patch, one mean, or one attention map.

The core question:

```text
Which distributional summary of patch states should survive?
```

Given:

```math
\mu_i
=
\frac{1}{n_i}\sum_j\delta_{u_{ij}},
```

distribution pooling computes:

```math
z_i
=
T(\mu_i).
```

## C/R/G/S Placement

```text
G:
    metric, kernel, transport cost, or prototype geometry

C:
    optional feature map, binning map, kernel map, or assignment map

R:
    CDF, quantile, histogram, MMD sketch, or transport-aware statistic

S:
    task loss if T is learned; unsupervised/statistical choice otherwise
```

Surviving statistic:

```math
z_i
=
T(\mu_i).
```

## Files

- `01_cdf_quantile_histogram_readouts.md`: empirical CDFs, quantiles, and
  histograms.
- `02_kernel_mmd_and_sketches.md`: kernel mean embeddings and finite sketches.
- `03_transport_and_geometry_aware_pooling.md`: OT and prototype geometry.
- `04_failure_modes.md`: sample complexity, lost layout, and metric mismatch.

