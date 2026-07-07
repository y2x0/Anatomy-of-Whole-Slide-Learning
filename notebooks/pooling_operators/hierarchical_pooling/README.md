# Hierarchical Pooling

Hierarchical pooling aggregates patches into regions, regions into slide
statistics, and sometimes multiple scales into one prediction.

The core question:

```text
At which scale should evidence be compressed?
```

## C/R/G/S Placement

```text
G:
    parent-child maps, region boundaries, image pyramid, or tissue graph

C:
    local context inside regions and optional cross-region context

R:
    region-to-slide pooling, scale-weighted pooling, or multiscale readout

S:
    slide labels, survival outcomes, or auxiliary region constraints
```

Surviving statistic:

```math
z_i
=
\psi
\left(
\mathcal{R}^{(0)}(H_i^{(0)}),
\ldots,
\mathcal{R}^{(L)}(H_i^{(L)})
\right).
```

## Files

- `01_region_to_slide_pooling.md`: bottom-up region aggregation.
- `02_scale_weighted_multiscale_readout.md`: combining evidence across scales.
- `03_hierarchical_attention_and_effective_weights.md`: effective patch weights
  through nested pooling.
- `04_failure_modes.md`: premature compression, bad regions, and scale mismatch.

## Anchor Ideas

- HIPT-style nested visual tokens.
- HACT-style cell-to-tissue graph hierarchy.
- Region-based MIL and pyramid-style WSI modeling.

