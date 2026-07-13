# Transformer MIL Score Credit and Deletion

## 1. Local Path Credit

For target score `F` and token `h_j`, a local gradient is

```math
g_j=\frac{\partial F}{\partial h_j}.
```

With attention edge `a_jell`, the derivative includes both value and routing
paths:

```math
\frac{\partial F}{\partial h_j}
=\sum_\ell
\frac{\partial F}{\partial a_{\ell j}}
\frac{\partial a_{\ell j}}{\partial h_j}
+\sum_\ell
\frac{\partial F}{\partial V_j}
\frac{\partial V_j}{\partial h_j}.
```

## 2. Token Deletion

Recompute the full transformer on a reduced bag:

```math
\Delta_j
=F(H)-F(H\setminus h_j).
```

This changes all softmax denominators and potentially positional layout.

## 3. Attention Approximation

For Nystrom or sampled attention, deletion may alter landmark selection or
sampled support. The explanation then includes approximation-selection effects.

## 4. Sign

Attention weights are nonnegative. Score credit and deletion effects are signed
and target-specific.

