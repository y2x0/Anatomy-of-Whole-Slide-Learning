# CTransPath: MoCo-Style Contrastive Learning with a Swin Encoder

## 1. Patch Object

CTransPath represents a histopathology image patch with a hierarchical Swin
Transformer encoder:

```math
h=f_\phi(x),
\qquad
u=\frac{g(h)}{\|g(h)\|_2}.
```

The windowed attention encoder changes the local context operator before the
contrastive objective is applied.

## 2. Queue-Based Contrast

For query `u_q`, positive key `u_k^+`, and a queue `Q` of negative keys,

```math
\ell_q
=-\log
\frac{\exp(u_q^{\top}u_k^+/\tau)}
{\exp(u_q^{\top}u_k^+/\tau)
+\sum_{v\in Q}\exp(u_q^{\top}v/\tau)}.
```

The momentum key encoder supplies slowly changing keys and the queue enlarges
the negative reference distribution.

## 3. Surviving Statistic

The objective preserves augmentation-invariant local morphology in angular
embedding geometry. It does not preserve slide coordinates unless coordinates
are included in the input or later slide aggregation.

## 4. Pathology Caveat

If two augmented views alter a rare diagnostic feature, the objective treats
that feature as nuisance. If stain or scanner is stable within a batch, it may
become a shortcut in the embedding geometry.

