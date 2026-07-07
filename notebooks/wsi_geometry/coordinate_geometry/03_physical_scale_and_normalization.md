# Physical Scale And Normalization

WSI coordinates are not automatically comparable across slides.

Raw pixel coordinates depend on:

```text
scanner resolution
magnification
downsampling level
tile size
tissue mask
cropping origin
rotation and flipping
```

Geometry becomes meaningful only after the coordinate frame is specified.

## Pixel Coordinates Versus Physical Coordinates

Let a patch coordinate in pixels be:

```math
c_{ij}^{\mathrm{px}}
=
(x_{ij}^{\mathrm{px}},y_{ij}^{\mathrm{px}}).
```

If the scanner resolution is $r_i$ microns per pixel, the physical coordinate
is:

```math
c_{ij}^{\mu m}
=
r_i c_{ij}^{\mathrm{px}}.
```

Distances should be compared in physical units:

```math
\|c_{ij}^{\mu m}-c_{ik}^{\mu m}\|_2
=
r_i\|c_{ij}^{\mathrm{px}}-c_{ik}^{\mathrm{px}}\|_2.
```

If $r_i$ changes across slides, fixed pixel radii do not represent fixed tissue
distances.

## Normalized Coordinates

A common normalization maps coordinates to a unit square:

```math
\tilde c_{ij}
=
\left(
\frac{x_{ij}-x_{\min}}{x_{\max}-x_{\min}},
\frac{y_{ij}-y_{\min}}{y_{\max}-y_{\min}}
\right).
```

This removes slide size but changes physical meaning:

```math
\|\tilde c_j-\tilde c_k\|_2
```

is a relative slide distance, not a tissue distance.

## Tissue-Mask Coordinates

Coordinates can be normalized over the tissue mask rather than the full slide:

```math
\Omega_i^{\mathrm{tissue}}
=
\{c:\text{patch at }c\text{ passes tissue threshold}\}.
```

This avoids background-dominated frames, but it makes location depend on tissue
segmentation.

## Scale Index

If patches are extracted at magnification level $\ell$, each coordinate has a
scale:

```math
(h_{ij}^{(\ell)},c_{ij}^{(\ell)},\Delta_\ell),
```

where $\Delta_\ell$ is the physical width of a patch at that level.

Two patches with the same coordinate but different $\Delta_\ell$ represent
different fields of view. A multiscale method should therefore model:

```math
G_i
=
\{(c_{ij}^{(\ell)},\Delta_\ell)\}_{j,\ell}.
```

## C/R/G/S Placement

```text
G:
    coordinate frame plus scale calibration

C:
    neighborhoods and positional encodings depend on physical or normalized distance

R:
    spatial summaries may be scale-specific

S:
    task supervision can accidentally reward scanner-specific coordinate frames
```

## Dense Summary

Coordinates are not just numbers. They are numbers in a frame.

```math
\text{pixel distance}
\ne
\text{physical distance}
\ne
\text{normalized slide distance}.
```

Many geometry failures are actually coordinate-frame failures.
