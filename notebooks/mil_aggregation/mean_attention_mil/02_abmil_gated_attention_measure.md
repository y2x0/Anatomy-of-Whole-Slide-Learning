# ABMIL Gated Attention Measure

## 1. Instance Scores

ABMIL computes a gated score for instance representation `h_ij`:

```math
s_{ij}
=w^{\top}
\left[
\tanh(Vh_{ij})
\odot
\mathrm{sigm}(Uh_{ij})
\right].
```

The normalized attention is

```math
a_{ij}
=\frac{\exp(s_{ij})}{\sum_\ell\exp(s_{i\ell})}.
```

## 2. Readout

```math
z_i=\sum_j a_{ij}h_{ij}.
```

The weights form a probability measure over instances, while the vector readout
is its first moment under the learned score geometry.

## 3. Weight Versus Evidence

`a_ij` is nonnegative and normalized within a bag. It is not a signed class
contribution. For linear class head `w_c`, the algebraic term is

```math
\gamma_{ijc}=a_{ij}w_c^{\top}h_{ij}.
```

For nonlinear heads, deletion or target-specific attribution is required.

## 4. Failure Modes

Softmax normalization makes attention relative to the bag. Adding an unrelated
patch can lower every existing weight; a stain artifact can win the competition;
and one patch can absorb nearly all mass without being causally necessary.

