# Perturbation Distributions And Off-Manifold Inputs

Primary anchors:

- Ribeiro, Singh, Guestrin. "Why Should I Trust You?" KDD 2016.
- Lundberg, Lee. "A Unified Approach to Interpreting Model Predictions."
  NeurIPS 2017.

## Perturbation Kernel

Let masking variable be:

```math
M
\in
\left\{
0,1
\right\}^{n}.
```

A perturbation operator creates:

```math
X^{(M)}
=
\mathcal{T}
\left(
X,M,B
\right),
```

where `B` specifies replacement values or samples. A perturbation explanation
estimates behavior under:

```math
M
\sim
q(M\mid X).
```

The result is conditional on both `T` and `q`.

## Local Surrogate

For interpretable mask vector `m`, LIME-style fitting solves:

```math
g^{\star}
=
\arg\min_{g\in\mathcal{G}}
\mathbb{E}_{M\sim q}
\left[
\pi_X(M)
\left(
F(X^{(M)})-g(M)
\right)^2
\right]
+
\Omega(g).
```

The explanation coefficients describe the best local surrogate under the
chosen perturbation and locality weights.

## Off-Manifold Risk

Let real WSI representations lie on support:

```math
\mathcal{M}
=
\mathrm{supp}
\left(
p_{\mathrm{WSI}}
\right).
```

Independent patch deletion or zero replacement can produce:

```math
X^{(M)}
\notin
\mathcal{M}.
```

Model behavior off support is unconstrained by training data. Large score
changes can reflect extrapolation rather than importance of removed tissue.

## Marginal And Conditional Missingness

Marginal replacement samples:

```math
X_S
\sim
p(X_S).
```

Conditional replacement samples:

```math
X_S
\sim
p
\left(
X_S
\mid
X_{\setminus S}
\right).
```

The first breaks feature dependence. The second preserves observational
dependence but may retain information about the removed component through its
correlates.

## WSI Removal Semantics

Removing a patch can mean:

```math
\text{delete the instance},
```

```math
\text{replace its embedding with zero},
```

```math
\text{replace it with benign tissue},
```

or:

```math
\text{inpaint spatially plausible tissue}.
```

These interventions change bag cardinality, attention normalization, tissue
distribution, or spatial context differently.

## Perturbation Identifiability

An explanation under one perturbation law identifies:

```math
\mathbb{E}_{q}
\left[
F(X)-F(X^{(M)})
\right],
```

not a perturbation-independent notion of importance.
