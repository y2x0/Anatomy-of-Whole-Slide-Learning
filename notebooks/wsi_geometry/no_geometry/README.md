# No Geometry

No-geometry models treat the slide as an unordered multiset of patch embeddings.

The geometry is:

```math
G_i
=
\varnothing.
```

The model sees:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i},
```

but not:

```math
C_i
=
\{c_{ij}\}_{j=1}^{n_i}.
```

## Files

- `01_permutation_invariance_as_geometry_erasure.md`: what is erased when coordinates disappear.
- `02_when_no_geometry_is_correct.md`: tasks where morphology prevalence is enough.
- `03_failure_modes.md`: arrangement blindness, shortcut safety, and false interpretability.

## C/R/G/S Placement

```text
G:
    empty

C:
    patchwise map, set attention, or complete-graph attention without physical edges

R:
    mean, max, attention, prototype, distribution, or other permutation-invariant readout

S:
    slide-level label, survival target, or retrieval objective
```

The central theorem is simple:

```math
f(H_i)
=
f(PH_i)
```

for any permutation
```math
P
```
of patch order. This is exactly right for some tasks and
catastrophically wrong for others.
