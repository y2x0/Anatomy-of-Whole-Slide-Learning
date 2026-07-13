# Invariance and Surviving Statistics

## 1. Invariance

An objective encourages invariance to transformation `T` when

```math
f_\phi(x)\approx f_\phi(Tx).
```

The useful question is whether `T` is nuisance or pathology signal.

## 2. Complementary Information

If a pretraining objective removes variation in subspace `U`, then a downstream
task depending on `U` faces an information bottleneck:

```math
Y\not\!\perp\!\!\!\perp U,
\qquad
f_\phi(X)\perp U\mid X\text{ after enforced invariance}.
```

The second relation is an idealized limit, but it describes the design risk.

## 3. Surviving Statistic by Objective

```text
contrastive: relative similarity among sampled views;
distillation: teacher-defined relations;
masked modeling: information predictable from visible context;
vision-language: caption- or knowledge-aligned semantics;
slide pretraining: arrangement and long-range co-occurrence if modeled;
retrieval: neighborhood structure under the learned metric.
```

No objective preserves “all pathology” without a definition of pathology and a
validation distribution.

