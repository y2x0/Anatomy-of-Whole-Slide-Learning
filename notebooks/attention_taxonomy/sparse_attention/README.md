# Sparse Attention

Sparse attention changes the normalization map so some weights can be exactly
zero.

```math
a_j
=
0
\quad
\text{for many }j.
```

This folder studies attention as support selection:

```text
01_sparsemax_projection.md
    sparsemax as Euclidean projection onto the simplex

02_entmax_and_alpha_normalization.md
    entmax as a bridge between softmax and sparsemax

03_topk_and_block_sparse_attention.md
    hard support constraints, top-k, windows, and block masks

04_failure_modes.md
    wrong support, brittle gradients, rare-pattern exclusion
```

