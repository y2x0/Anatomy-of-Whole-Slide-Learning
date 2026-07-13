# Correlated Instance Context

## 1. Set MIL Limitation

A context-free readout computes

```math
z_i=\mathcal R(\{h_{ij}\}_j),
```

so each instance is processed without explicit pairwise interaction before
aggregation. TransMIL treats correlated instances as a first-class object.

## 2. Self-Attention Context

For token matrix `H`,

```math
Q=HW_Q,
\qquad
K=HW_K,
\qquad
V=HW_V,
```

```math
\widetilde H
=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V.
```

Each contextualized token depends on every supported token.

## 3. Surviving Statistic

The readout no longer receives independent patch states. It receives relational
states:

```math
z_i=\mathcal R(\widetilde H_i),
\qquad
\widetilde h_{ij}=\sum_\ell a_{ij\ell}h_{i\ell}W_V.
```

The surviving statistic contains pairwise similarity-modulated information,
subject to the final readout bottleneck.

## 4. Failure

Self-attention can amplify correlated nuisance patches. A high attention edge
explains routing inside the context operator, not automatically final class
evidence.

