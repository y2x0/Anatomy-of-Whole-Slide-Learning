# Sparsemax Projection

Softmax maps every finite score to positive mass. Sparsemax maps scores to the
closest point on the simplex.

Given:

```math
s
\in
\mathbb{R}^{n},
```

sparsemax is:

```math
\mathrm{sparsemax}(s)
=
\arg\min_{p\in\Delta^{n-1}}
\frac{1}{2}
\|p-s\|_2^2.
```

The simplex is:

```math
\Delta^{n-1}
=
\left\{
p\in\mathbb{R}^{n}:
p_j\ge 0,
\quad
\sum_jp_j=1
\right\}.
```

## Closed Form

The solution has the form:

```math
p_j
=
\max(s_j-\tau,0),
```

where `tau` is chosen so the weights sum to one:

```math
\sum_j\max(s_j-\tau,0)
=
1.
```

The active support is:

```math
S
=
\{j:s_j>\tau\}.
```

For `j` outside `S`:

```math
p_j=0.
```

## Attention Output

Sparse attention output is:

```math
m
=
\sum_{j\in S}p_jv_j.
```

The readout is still a weighted first moment, but only over an active support.

## Gradient On Active Support

For active indices:

```math
\frac{\partial p_j}{\partial s_k}
=
\mathbf{1}\{j=k\}
-
\frac{1}{|S|}
\qquad
j,k\in S.
```

If either index is inactive:

```math
\frac{\partial p_j}{\partial s_k}
=
0
```

except at support-changing boundaries.

Thus sparsemax creates exact zeros and zero gradients for excluded tokens.

## WSI Interpretation

Sparsemax attention can mean:

```text
ignore most patches exactly
pool only a learned subset
make attention maps visually cleaner
```

But the mathematical cost is support brittleness. A patch outside the active set
does not influence the output or gradient until its score crosses the threshold.

## Dense Summary

Sparsemax replaces:

```math
\mathrm{softmax}(s)
```

with:

```math
\mathrm{proj}_{\Delta}(s).
```

The inductive bias is:

```text
only a subset of instances should survive the attention readout
```

This is useful for sparse-positive MIL, but dangerous when weak evidence is
distributed across tissue.

