# Grid Geometry

Grid geometry treats the slide as a regular or sparse lattice.

A full image is naturally indexed by:

```math
(r,s)\in\mathbb{Z}^{2}.
```

But a tiled WSI after tissue masking is usually a sparse grid:

```math
\Lambda_i
\subset
\mathbb{Z}^{2}.
```

Each occupied grid site has a feature:

```math
h_i(r,s)
\in
\mathbb{R}^{d},
\qquad
(r,s)\in\Lambda_i.
```

## Files

- `01_slide_as_sparse_lattice.md`: WSI as an incomplete grid.
- `02_windows_convolutions_and_shifted_context.md`: local grid neighborhoods.
- `03_aliasing_masks_and_failure_modes.md`: missing tissue, aliasing, and window artifacts.

## C/R/G/S Placement

```text
G:
    lattice index and tissue mask

C:
    convolution, windowed attention, shifted windows, or grid scan

R:
    pooling over sites, windows, or multiscale grids

S:
    task loss plus any augmentation or scale-consistency assumptions
```
