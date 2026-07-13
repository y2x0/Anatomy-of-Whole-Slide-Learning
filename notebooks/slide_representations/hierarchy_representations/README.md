# Hierarchy Representations

A hierarchy representation treats a slide as nested tissue units rather than a
flat bag.

The core question:

```text
How do patch, region, and slide scales relate?
```

A hierarchical slide object has levels:

```math
\mathcal{X}_i
=
\left(
V_i^{(0)},V_i^{(1)},\ldots,V_i^{(L)},
\pi_i^{(0)},\ldots,\pi_i^{(L-1)}
\right),
```

where
```math
V_i^{(\ell)}
```
 is the set of units at scale
```math
\ell
```
, and
```math
\pi_i^{(\ell)}
```
maps lower-scale units to parent units.

## Files

- `01_slide_as_hierarchy.md`: nested tissue units and parent maps.
- `02_coarsening_and_multiscale_context.md`: bottom-up, top-down, and lateral
  context.
- `03_hierarchical_readout.md`: multiscale readout and surviving statistics.
- `04_failure_modes.md`: boundary errors, premature compression, and scale
  mismatch.

## Anchor Methods

HIPT and HACT are canonical examples:

```text
HIPT:
    image pyramid and nested visual tokens

HACT:
    cell-to-tissue hierarchical graph
```

The unifying idea is not the specific architecture. It is the choice to make the
slide a structured multiscale object before prediction.
