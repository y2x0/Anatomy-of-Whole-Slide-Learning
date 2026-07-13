# Transformer MIL Complexity and Sampling

## 1. Exact Cost

For `n` tokens and hidden width `d`, attention has approximate cost

```math
\mathcal O(n^2d)
```

and memory cost dominated by the `n` by `n` attention matrix.

## 2. Sampling as a Readout Bias

If a sampler retains subset `I_i`, the effective bag is

```math
B_i^{\mathrm{sample}}=\{h_{ij}:j\in I_i\}.
```

The model estimates

```math
F(B_i^{\mathrm{sample}})
```

not necessarily `F(B_i)`. Sampling changes the instance distribution and can
remove rare positive evidence before context is computed.

## 3. Complexity Versus Support

Landmark, sparse, windowed, and hierarchical attention trade support for cost.
The correct comparison is not only runtime; it is which pairwise relations and
which rare patches survive.

