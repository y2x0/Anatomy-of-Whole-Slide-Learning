# Attention Support and Long-Range Interactions

## 1. Dense Support

Exact self-attention has support

```math
\mathrm{supp}(A_i)
=\{(j,\ell):1\le j,\ell\le n_i\}.
```

Every token can influence every other token in one layer, but the magnitude of
that influence is controlled by the learned kernel.

## 2. Effective Support

Define thresholded support

```math
E_\epsilon
=\{(j,\ell):a_{j\ell}>\epsilon\}.
```

The dense matrix may have a sparse effective support. A heatmap or attention
visualization should report the threshold and layer.

## 3. Long-Range Loss

Approximation, sampling, or masking can remove weak but collectively important
long-range interactions. A single high edge is not proof that all relevant
context is local.

## 4. WSI Test

Permute only distant coordinate groups while preserving local neighborhoods. A
model whose output changes is using long-range or order-dependent information;
the change can be helpful geometry or a shortcut.

