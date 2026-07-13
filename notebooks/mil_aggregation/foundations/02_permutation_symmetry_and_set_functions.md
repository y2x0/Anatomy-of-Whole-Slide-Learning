# Permutation Symmetry and Set Functions

## 1. Invariance Requirement

For a bag represented as an unordered multiset, every permutation `pi` should
satisfy

```math
F(\{h_1,\ldots,h_n\})
=F(\{h_{\pi(1)},\ldots,h_{\pi(n)}\}).
```

This is invariance of the bag prediction. Instance-wise context may be
equivariant rather than invariant:

```math
\widetilde h_{\pi(j)}(\pi H)
=\widetilde h_j(H).
```

## 2. Deep Sets Form

A canonical invariant architecture is

```math
F(B)=\rho\left(\sum_{h\in B}\phi(h)\right).
```

Mean pooling is the normalized special case. Max pooling retains an extreme
statistic. Attention replaces the fixed sum weights with learned, normalized
weights.

## 3. Geometry Breaks Pure Set Semantics

If coordinates enter the representation,

```math
F(B,C)
```

is invariant to jointly permuting `(h_j,c_j)` but not to arbitrarily permuting
features while holding coordinates fixed. A graph or sequence adds a relation
operator before readout.

## 4. Expressivity Is Not Identifiability

Two different context/readout decompositions can implement the same invariant
function:

```math
\mathcal R_1(\mathcal C_1(H))
=\mathcal R_2(\mathcal C_2(H)).
```

Observed bag predictions alone cannot identify where the model stored the
interaction. Explanations must inspect the actual forward map or interventions.

