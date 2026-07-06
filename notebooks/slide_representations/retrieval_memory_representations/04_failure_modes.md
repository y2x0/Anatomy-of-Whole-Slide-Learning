# Retrieval Memory Failure Modes

Retrieval systems fail differently from ordinary slide classifiers because the
database is part of the model.

## 1. Cohort Leakage

If train and test slides from the same patient or case family both enter memory,
retrieval can become memorization:

```math
\operatorname{sim}(z_i,z_k)\approx1
```

because of patient, block, scanner, or near-duplicate structure.

The evaluation split must be patient-level or institution-aware when relevant.

## 2. Metric Shortcut

The nearest neighbors are nearest under the chosen metric:

```math
k^\star
=
\operatorname*{argmax}_{k}
\operatorname{sim}(z_i,z_k).
```

If $z$ encodes stain, scanner, tissue amount, or organ more strongly than
disease mechanism, retrieval follows those shortcuts.

The model may retrieve:

```text
visually similar nuisance
```

instead of:

```text
clinically similar disease
```

## 3. Memory Drift

A memory bank built at time $t_0$:

```math
\mathcal{M}_{t_0}
```

may not match data at time $t_1$:

```math
p_{t_0}(z,y)\ne p_{t_1}(z,y).
```

New scanners, stains, protocols, organs, or diagnostic categories can change
the retrieval neighborhood.

## 4. False Authority

Retrieved cases are persuasive. But:

```math
k\in\mathcal{N}_K(i)
```

only means close in embedding space. It does not mean the case is clinically
analogous or correctly labeled.

Retrieval explanations can be wrong while looking plausible.

## 5. Hubness

In high-dimensional spaces, some database points become nearest neighbors for
many queries:

```math
\#\{i:k\in\mathcal{N}_K(i)\}
\gg
\frac{KN}{N}
=K.
```

These hubs can dominate prediction and retrieval interpretation.

## 6. Non-Stationary Label Semantics

If reports or diagnoses are used as memory values, label meaning may vary across
institutions or time:

```math
y_k
\not\equiv
y_{k'}
```

even when the string is identical. Diagnostic criteria and reporting styles can
change.

## Diagnostic Questions

1. What is stored in memory?
2. Are keys patch-level, slide-level, text-level, or multimodal?
3. What similarity metric defines neighbors?
4. Is evaluation patient-level and leakage-safe?
5. Are retrieved cases clinically or only visually similar?
6. How is memory updated when the data distribution shifts?

## Dense Summary

The retrieval assumption is:

```text
similar embeddings should imply useful precedent
```

It fails when:

```text
embedding similarity tracks nuisance, leakage, stale labels, or archive artifacts
```
