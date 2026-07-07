# Coordinate Geometry

Coordinate geometry treats a slide as a marked point cloud:

```math
X_i
=
\{(h_{ij},c_{ij})\}_{j=1}^{n_i},
\qquad
c_{ij}\in\mathbb{R}^{2}.
```

Coordinates may enter through absolute position, relative displacement,
distance, sinusoidal encodings, radial features, or coordinate-aware attention.

## Files

- `01_coordinates_as_marks.md`: marked point clouds and coordinate-dependent maps.
- `02_relative_displacements_and_positional_encodings.md`: absolute versus relative geometry.
- `03_physical_scale_and_normalization.md`: microns, magnification, resizing, and coordinate frames.
- `04_failure_modes.md`: shortcut coordinates, scale mismatch, and boundary artifacts.

## C/R/G/S Placement

```text
G:
    coordinate set C_i

C:
    patch context can use c_j, c_j - c_k, or distance features

R:
    readout can weight patches by spatial position or region

S:
    task loss may reward spatial correlations, even if they are shortcuts
```
