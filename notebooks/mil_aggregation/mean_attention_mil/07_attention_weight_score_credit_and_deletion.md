# Attention Weight, Score Credit, and Deletion

## 1. Three Quantities

For patch `j`, distinguish

```math
a_{ij},
\qquad
\gamma_{ijc},
\qquad
\Delta_{ijc}=F_{ic}(B_i)-F_{ic}(B_i\setminus j).
```

They are routing weight, score credit, and deletion effect.

## 2. When They Coincide

Coincidence requires restrictive conditions: a fixed attention vector, linear
readout and head, and a deletion baseline that removes only the corresponding
additive term. Softmax recomputation violates these conditions because

```math
a_{i\ell}^{(-j)}
\ne a_{i\ell}
```

for every remaining patch `ell` in general.

## 3. Target Dependence

For class, survival risk, hazard horizon, or retrieval score, the same weights
can yield different signed credits:

```math
\gamma_{ij}^{(q)}
=a_{ij}\left(
\frac{\partial F_q}{\partial z_i}
\right)^{\top}h_{ij}
```

when the head is locally linear around the current representation.

## 4. Reporting Rule

Never label an attention map “evidence” without naming whether it is weight,
credit, or deletion effect.
