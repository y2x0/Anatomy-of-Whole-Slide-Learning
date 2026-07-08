# Softmax Attention

Softmax attention is the default normalization for learned weighting.

```math
a_j
=
\frac{\exp(s_j)}
{\sum_{\ell=1}^{n}\exp(s_\ell)}.
```

This folder studies what the softmax does mathematically:

```text
01_attention_as_kernel_smoother.md
    attention as a normalized exponential kernel

02_softmax_geometry_temperature_entropy.md
    temperature, entropy, effective support, and gradients

03_attention_pooling_vs_context_attention.md
    readout attention versus token-to-token context attention

04_failure_modes.md
    dense support, denominator competition, collapse, and dilution
```

