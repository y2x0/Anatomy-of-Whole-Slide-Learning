# State Space Context With MIL Readouts

## 1. Ordered context

A state-space MIL model receives an ordered sequence

```math
H_i=(h_{i1},\ldots,h_{in_i}),
```

and updates a hidden state

```math
s_{ij}
=
\overline A_j s_{i,j-1}
+
\overline B_j h_{ij},
\qquad
o_{ij}
=
C_j s_{ij}+D h_{ij}.
```

For a fixed linear SSM, the coefficients are shared or position-dependent. For
a selective state space, they can depend on h_{ij}. Either way, the sequence
operator is not invariant to arbitrary patch permutations.

## 2. MambaMIL followed by attention

A downstream attention readout can be written

```math
a_{ij}
=
\frac{\exp q(o_{ij})}{\sum_k\exp q(o_{ik})},
\qquad
z_i
=
\sum_j a_{ij}o_{ij}.
```

The model is state-space context plus attention readout. The weights a_{ij}
select states that already summarize prefixes or reordered segments.

If a state o_{ij} has receptive field

```math
o_{ij}
=
f(h_{i1},\ldots,h_{ij}),
```

then attention at j is not a local patch score. It is a score for a prefix
summary.

## 3. SR-Mamba reordering

SR-Mamba applies a sequence reordering operator before or inside the state-space
blocks. Let rho_i map patch states to a sequence order:

```math
H_i^{\mathrm{ord}}
=
\rho_i(H_i,X_i),
```

where X_i may be coordinates or feature-derived grouping. The output is restored
through rho_i^{-1} when the architecture requires patch-aligned states:

```math
O_i
=
\rho_i^{-1}
\left(
\mathrm{SSM}
\left(
\rho_i(H_i,X_i)
\right)
\right).
```

The reordering is part of G and C, not the readout. A permutation of the input
that leaves the physical slide unchanged should also transform rho equivariantly
if the model is to avoid accidental index dependence.

## 4. Readout alternatives

After state-space context, one may use

```math
z_i^{\mathrm{mean}}
=
\frac{1}{n_i}\sum_j o_{ij},
\qquad
z_i^{\mathrm{max}}
=
\max_j o_{ij},
```

or an attention statistic

```math
z_i^{\mathrm{att}}
=
\sum_j a_{ij}o_{ij}.
```

The mean is invariant to output order but not to the order-dependent context that
generated o. Therefore a permutation-invariant readout does not restore
permutation invariance to the full model.

A scalar state-space summary can also be used directly:

```math
z_i=o_{in_i}.
```

This is an endpoint statistic with maximal order dependence and no explicit
coverage guarantee for late or early positive evidence.

## 5. Hybrid comparison

Two models can share the same attention readout but have different context:

```math
z_i^{\mathrm{ABMIL}}
=
R_{\mathrm{att}}(H_i),
\qquad
z_i^{\mathrm{Mamba+att}}
=
R_{\mathrm{att}}(\mathrm{SSM}(H_i)).
```

The second can represent sequential interactions unavailable to the first, but
only relative to the chosen order. The readout formula alone does not identify
the model's inductive bias.

## 6. C/R/G/S placement

```text
C: fixed or selective state-space recurrence, including SR-Mamba reordering.

R: endpoint, mean, max, or attention over state outputs.

G: a chosen patch order, coordinate order, feature-derived order, or segment
   permutation.

S: slide classification or survival labels; sequence pretraining is an upstream
   objective if used.

The surviving statistic is a summary of an order-conditioned dynamical system,
even when the final pooling operator is symmetric.
```
