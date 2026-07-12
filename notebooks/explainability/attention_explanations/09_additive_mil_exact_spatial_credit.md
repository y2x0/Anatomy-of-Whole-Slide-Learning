# Additive MIL Exact Spatial Credit

Primary anchor:

- Javed et al. "Additive MIL: Intrinsically Interpretable Multiple Instance
  Learning." NeurIPS 2022 Workshop.
  https://arxiv.org/abs/2206.01794

## Composition Change

Conventional attention MIL applies the predictor after aggregation:

```math
F_{\mathrm{attn}}(X)
=
\psi
\left(
\sum_i
\alpha_i m_i(x_i)
\right).
```

Additive MIL moves a patch predictor inside the sum:

```math
F_{\mathrm{add}}(X)
=
\sum_i
\psi_p
\left(
m_i(x_i)
\right).
```

For class `c`, define patch evidence:

```math
e_{i,c}
=
\psi_{p,c}
\left(
m_i(x_i)
\right).
```

Then:

```math
F_c(X)
=
\sum_i e_{i,c}.
```

## Exact Removal Credit

For bag with patch `i` removed:

```math
F_c(X_{-i})
=
\sum_{j\ne i}
e_{j,c}.
```

Therefore:

```math
F_c(X)-F_c(X_{-i})
=
e_{i,c}.
```

This is exact finite score credit, not a local derivative or attention proxy.

## Signed And Class-Specific Evidence

```math
e_{i,c}>0
```

supports class `c`, while:

```math
e_{i,c}<0
```

opposes it. Different classes can receive different signs from the same patch.

## Completeness

With empty-bag baseline score zero:

```math
\sum_i e_{i,c}
=
F_c(X)-F_c(\varnothing).
```

If a bias `b_c` is present:

```math
F_c(X)
=
b_c
+
\sum_i e_{i,c},
```

the bias is baseline credit rather than patch credit.

## Probability Is Not Additive

For binary probability:

```math
p(X)
=
\sigma
\left(
F(X)
\right),
```

patch probability effect is:

```math
p(X)-p(X_{-i})
=
\sigma(F)
-
\sigma(F-e_i).
```

Exact additivity holds in score or log-odds space, not probability space.

## Interaction Cost

The additive score has zero cross-instance second derivative when patch
evidence is independently computed:

```math
\frac{\partial^2F}
{\partial x_i\partial x_j}
=
0
\qquad
i\ne j.
```

Exact credit is obtained by restricting explicit final-score interactions.
Context can be introduced before evidence computation, but then patch evidence
may depend on other patches and removal semantics must be reconsidered.
