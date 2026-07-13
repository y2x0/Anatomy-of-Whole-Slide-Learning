# Masked Modeling and Reconstruction

## 1. Masked Input

Partition a patch or image into tokens `x_1,...,x_N` and mask set `M`:

```math
\widetilde x_j
=\begin{cases}
\mathrm{mask},&j\in M,\\
x_j,&j\notin M.
\end{cases}
```

An encoder-decoder predicts the hidden content:

```math
\widehat x_M=D_\psi(E_\phi(\widetilde x)).
```

## 2. Reconstruction Objective

```math
\mathcal L_{\mathrm{MIM}}
=\frac{1}{|M|}
\sum_{j\in M}
d(\widehat x_j,x_j).
```

The target can be pixels, token features, or a teacher representation. The
choice determines whether the model must preserve appearance, semantics, or a
teacher-defined embedding.

## 3. Masking as an Inductive Bias

Large masks force context-based prediction; small masks permit local texture
interpolation. In pathology,

```text
mask size -> local texture versus tissue context;
mask shape -> token independence versus region structure;
target space -> pixels versus semantics;
sampling -> common tissue versus rare morphology.
```

Phikon and related masked-image-modeling approaches should therefore be
compared by mask and target geometry, not only by encoder size.

## 4. Failure Mode

If reconstruction is dominated by stain or scanner statistics, low loss can
coexist with poor biological transfer. A reconstruction objective is not a
clinical sufficiency guarantee.

