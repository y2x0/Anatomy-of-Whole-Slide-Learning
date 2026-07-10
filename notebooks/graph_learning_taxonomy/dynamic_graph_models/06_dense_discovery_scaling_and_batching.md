# Dense Discovery, Scaling, And One-Slide Batching

This note isolates the computational boundary of WiKG graph construction: a sparse selected graph discovered through a dense score matrix and a released batching path specialized to one WSI.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Complexity

The output graph has only:

```math
|E|
=
nK
```

directed selected edges, but the released construction first materializes all
pair scores:

```math
S
\in
\mathbb{R}^{n\times n}.
```

The dense score cost is:

```math
\mathrm{time}
=
\mathcal{O}
\left(
n^2d
\right),
```

```math
\mathrm{score\ memory}
=
\mathcal{O}
\left(
n^2
\right).
```

Selected edge states cost:

```math
\mathcal{O}
\left(
nKd
\right)
```

additional memory.

For `n=10,000`, the dense score matrix contains:

```math
10^8
```

entries, which is approximately 400 MB in 32-bit floating point before
accounting for gradients, temporary tensors, head-tail features, and selected
edge states. Sparse output does not imply sparse construction.

## 15. Batch Structure Is Specialized To One Slide

The paper reports batch size one. The released forward pass removes the leading
dimension before graph readout:

```text
h.squeeze(0)
```

and supplies no graph-membership vector to pooling. The implementation is
therefore specialized to one WSI per forward pass. A multi-slide batch requires
explicit padding and masks or a concatenated-node representation with graph
membership indices.

Without that change, variable-size slides cannot simply be stacked and pooled
as independent graphs.
