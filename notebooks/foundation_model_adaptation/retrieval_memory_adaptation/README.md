# Retrieval Memory Adaptation

Retrieval-memory adaptation changes prediction by changing the memory bank,
metric, prototypes, or retrieved context rather than necessarily changing model
weights.

For query slide embedding $z_i$ and memory:

```math
\mathcal{M}
=
\{(m_k,y_k)\}_{k=1}^{K},
```

prediction depends on neighbors:

```math
\mathcal{N}_K(i)
=
\mathrm{TopK}_{k}
\mathrm{sim}(z_i,m_k).
```

## Files

- `01_memory_bank_as_adaptation.md`: memory as trainable or curated task state.
- `02_knn_and_prototype_decision_rules.md`: nonparametric and prototype heads.
- `03_retrieval_augmented_heads.md`: retrieved context as conditioning.
- `04_failure_modes.md`: memory bias, metric mismatch, leakage, and stale banks.

## C/R/G/S Placement

```text
G:
    memory graph or nearest-neighbor topology

C:
    retrieved neighbors condition the query representation or head

R:
    readout may be a weighted vote, prototype score, or retrieval-conditioned head

S:
    labels enter through stored examples, prototype construction, or metric learning
```

## Dense Summary

Retrieval adaptation asks whether the model can stay mostly fixed while the
task lives in the memory.
