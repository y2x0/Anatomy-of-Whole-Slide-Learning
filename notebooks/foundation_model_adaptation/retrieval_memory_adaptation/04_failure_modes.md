# Failure Modes

## Memory Bias

The memory bank represents a distribution:

```math
P_{\mathcal{M}}(Z,Y).
```

If:

```math
P_{\mathcal{M}}(Z,Y)
\ne
P_{\mathrm{target}}(Z,Y),
```

retrieval predictions inherit memory bias.

## Metric Mismatch

Retrieval assumes:

```math
\mathrm{sim}(z_i,m_k)
\text{ high}
\Rightarrow
Y_i
\approx
Y_k.
```

If similarity captures organ, stain, scanner, or background instead of the task,
nearest neighbors are misleading.

## Leakage

If memory contains near-duplicates from the same patient or case:

```math
k\in\mathcal{M}
\quad
\text{same patient as query},
```

retrieval performance can overestimate generalization.

## Stale Memory

If the encoder is updated:

```math
F_{\phi_0}
\to
F_{\phi_t},
```

but memory embeddings are not recomputed, then query and memory live in
different spaces:

```math
z_i=F_{\phi_t}(X_i),
\qquad
m_k=F_{\phi_0}(X_k).
```

## Dense Summary

Retrieval adaptation is only as good as:

```text
memory coverage
metric alignment
leakage control
embedding consistency
```

It can look extremely strong while merely finding familiar cases.
