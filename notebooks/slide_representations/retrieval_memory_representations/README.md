# Retrieval Memory Representations

A retrieval-memory representation treats a slide as a query into an external
database of slides, patches, reports, labels, or prototypes.

The core question:

```text
What does this slide retrieve?
```

Instead of learning only:

```math
\widehat y_i=\mathcal{H}(z_i),
```

we use:

```math
\widehat y_i
=
\mathcal{H}(z_i,\mathcal{M}(z_i)),
```

where
```math
\mathcal{M}
```
is a memory or retrieval operator.

## Files

- `01_slide_as_query.md`: query embeddings and nearest-neighbor slide objects.
- `02_memory_banks_and_similarity.md`: indexed databases, keys, values, and
  metrics.
- `03_retrieval_augmented_prediction.md`: nonparametric and hybrid prediction.
- `04_failure_modes.md`: cohort leakage, metric shortcuts, and memory drift.

## Anchor Methods

Yottixel and SISH are canonical WSI retrieval systems. Foundation-model retrieval
methods extend the same idea with pretrained patch, slide, or multimodal
embeddings.

The representation is not just:

```text
slide embedding
```

but:

```text
slide embedding plus its neighborhood in memory
```
