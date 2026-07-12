# Faithfulness, Completeness, And Sensitivity

Primary anchors:

- Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML
  2017.
- Lundberg, Lee. "A Unified Approach to Interpreting Model Predictions."
  NeurIPS 2017.

## Completeness

For baseline `X_0`, additive attributions are complete if:

```math
\sum_{j=1}^{m}
\phi_j
=
F(X)
-
F(X_0).
```

Completeness conserves score difference. It does not ensure that individual
credits correspond to causal or human-recognizable factors.

## Integrated-Gradient Completeness

For straight path and differentiable `F`:

```math
\sum_j
\left(
X_j-X_{0,j}
\right)
\int_0^1
\frac{
\partial F
\left(
X_0+t(X-X_0)
\right)
}{
\partial X_j
}
\,\mathrm{d}t
=
F(X)-F(X_0).
```

This follows from the fundamental theorem for line integrals.

## Sensitivity

If inputs differ in one component and scores differ:

```math
X_{-j}=X_{-j}',
\qquad
F(X)\ne F(X'),
```

sensitivity requires nonzero attribution to component `j` under the relevant
baseline construction.

## Dummy Property

If `F` does not depend on component `j`:

```math
F(X)
=
F(X_{-j},X_j')
\qquad
\forall X_j',
```

then:

```math
\phi_j=0.
```

## Implementation Invariance

If two networks compute the same function:

```math
F(X)=G(X)
\qquad
\forall X,
```

an implementation-invariant method requires:

```math
\mathcal{E}(F,X)
=
\mathcal{E}(G,X).
```

Methods using internal activations or attention can violate this even when
predictions are identical.

## Infidelity

For perturbation `I` and attribution vector `Phi`, define:

```math
\mathrm{Infid}
\left(
\Phi,F,X
\right)
=
\mathbb{E}_{I}
\left[
\left(
I^{\top}\Phi
-
\left[
F(X)-F(X-I)
\right]
\right)^2
\right].
```

This metric is relative to the perturbation law. A low value under unrealistic
perturbations does not establish clinical faithfulness.

## Sufficiency And Comprehensiveness

For selected patch set `S`:

```math
\mathrm{Suff}(S)
=
F(X)
-
F(X_S),
```

```math
\mathrm{Comp}(S)
=
F(X)
-
F(X_{\setminus S}).
```

Small sufficiency gap means selected evidence preserves the score. Large
comprehensiveness means removing it reduces the score. Neither proves the set
is pathologically correct.
