# TransMIL Self-Attention and Nystrom Approximation

## 1. Exact Attention

For `n` patch tokens, exact self-attention forms an `n` by `n` kernel:

```math
A=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right).
```

The memory and compute cost are quadratic in `n`.

## 2. Landmark Approximation

Nystrom-style attention selects `m` landmarks with `m` much smaller than `n`:

```math
\widetilde A
\approx
A_{n\times m}A_{m\times m}^{\dagger}A_{m\times n}.
```

The approximation changes the effective interaction kernel and therefore the
context statistic, not only the runtime.

## 3. Approximation Error

If the true kernel is `A` and approximation is `A_tilde`, a downstream bound
depends on

```math
\|\widetilde H-AH\|
\le
\|\widetilde A-A\|\,\|H\|.
```

Small kernel error is not guaranteed by small landmark count; it depends on
landmark coverage and attention spectrum.

