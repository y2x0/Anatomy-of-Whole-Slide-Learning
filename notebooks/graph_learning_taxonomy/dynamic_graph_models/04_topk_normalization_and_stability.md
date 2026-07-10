# Top-K Normalization And Stability

This note separates the paper's printed normalization from the released computation, then derives sparse support and the piecewise differential structure of hard top-k selection.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Paper Equation Versus Released Computation

The paper says that a softmax quantifies head-tail similarity, but its printed
Equation 2 is:

```math
\omega_{vu}^{\mathrm{printed}}
=
\frac{q_vt_u^{\top}}
{\sum_{a=1}^{n}q_vt_a^{\top}}.
```

This expression is not a softmax. It may be negative, its denominator may be
zero, and it is not invariant to adding a rowwise constant to all logits.

The paper's pseudocode and released implementation instead do the following:

```math
s_{vu}
=
\frac{q_vt_u^{\top}}{\sqrt d},
```

```math
\mathcal{N}_K(v)
=
\mathrm{TopKIndices}
\left(
\{s_{va}\}_{a=1}^{n}
\right),
```

```math
\omega_{vu}
=
\frac{\exp(s_{vu})}
{\sum_{a\in\mathcal{N}_K(v)}\exp(s_{va})},
\qquad
u\in\mathcal{N}_K(v).
```

These notes use this implementation-faithful definition. Because the
exponential is strictly increasing, selecting top-k before or after a full
softmax gives the same support in exact arithmetic. Normalizing only after
selection changes the retained weights because discarded mass is removed.

## Sparse Directed Support

Define:

```math
A[v,u]
=
\mathbb{1}
\left\{
u\in\mathcal{N}_K(v)
\right\}.
```

The normalized selected-neighbor matrix is:

```math
\Omega[v,u]
=
\begin{cases}
\omega_{vu},
&
u\in\mathcal{N}_K(v),
\\
0,
&
u\notin\mathcal{N}_K(v).
\end{cases}
```

Every nonempty row sums to one:

```math
\Omega\mathbf{1}_n
=
\mathbf{1}_n.
```

But column sums are unconstrained:

```math
\mathbf{1}_n^{\top}\Omega
\ne
\mathbf{1}_n^{\top}
```

in general. Some patches can be selected as sources by many targets, while
others may be selected by none.

## Piecewise Differentiability Of Top-K

For target `v`, suppose the `K`th and `(K+1)`th largest logits satisfy a strict
margin:

```math
\gamma_v
=
s_{v,(K)}-s_{v,(K+1)}
>
0.
```

Small perturbations that preserve this ordering leave the selected set fixed.
Inside that region, selected weights have the standard softmax Jacobian:

```math
\frac{\partial\omega_{vu}}
{\partial s_{va}}
=
\omega_{vu}
\left(
\mathbb{1}\{u=a\}-\omega_{va}
\right),
\qquad
u,a\in\mathcal{N}_K(v).
```

For an unselected index `a`, the local derivative through the top-k operation
is zero while the support remains unchanged:

```math
\frac{\partial\omega_{vu}}
{\partial s_{va}}
=
0,
\qquad
a\notin\mathcal{N}_K(v).
```

At a rank boundary:

```math
s_{v,(K)}
=
s_{v,(K+1)},
```

the discrete support can switch. The graph map is piecewise smooth but not
globally smooth.

The margin therefore measures local topology stability. If:

```math
\gamma_v
\approx
0,
```

small feature or parameter perturbations can replace an edge.
