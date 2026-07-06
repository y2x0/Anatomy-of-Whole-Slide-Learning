# Unifying View

The slide representation notes should not be read as separate model catalogs.
They are different ways of specifying structure before aggregation.

This folder asks:

```text
What changes mathematically when the slide object changes?
```

The common template is:

```math
H_i
\xrightarrow{\operatorname{structure}}
\mathcal{X}_i
\xrightarrow{\mathcal{C}}
\widetilde{H}_i
\xrightarrow{\mathcal{R}}
z_i
\xrightarrow{\mathcal{H}}
\widehat{y}_i.
```

where:

```text
H_i:
    patch embeddings

structure:
    set, order, graph, hierarchy, distribution, memory, or latent geometry

mathcal{C}:
    context operator

mathcal{R}:
    readout operator

mathcal{H}:
    task head
```

## Notes

1. `01_crgs_placement_for_representations.md`
2. `02_adjacency_as_context_operator.md`
3. `03_paper_placement_matrix.md`
4. `04_extended_family_placement.md`

## Core Claim

Most whole-slide learning architectures differ by:

```text
which structure they put on patches
which context edges they allow
which statistics survive readout
which external geometry or memory is inherited
```

That is why representation and aggregation must be separated.
