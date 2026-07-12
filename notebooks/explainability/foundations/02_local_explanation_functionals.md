# Local Explanation Functionals

Primary anchors:

- Ribeiro, Singh, Guestrin. "Why Should I Trust You?" KDD 2016.
- Lundberg, Lee. "A Unified Approach to Interpreting Model Predictions."
  NeurIPS 2017.
- Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML
  2017.

## Explanation As A Functional

For scalar target `F(X)`, a local explanation is:

```math
\Phi
\left[
F;
X,
X_0,
\nu,
\mathcal{G}
\right],
```

where `X_0` is a reference, `nu` is a locality or perturbation distribution,
and `G` is the explanation class.

Changing any argument can change the explanation without changing `F` or `X`.

## Additive Attribution

For components `1,...,m`, an additive explanation has:

```math
g(z')
=
\phi_0
+
\sum_{j=1}^{m}
\phi_j z'_j,
```

with interpretable presence vector:

```math
z'
\in
\left\{
0,1
\right\}^{m}.
```

The coefficients are local credits under a specified missingness semantics.

## Gradient Functional

Local differential attribution is:

```math
\Phi_j^{\mathrm{grad}}
\left[
F;X
\right]
=
\frac{\partial F(X)}
{\partial X_j}.
```

It estimates infinitesimal sensitivity, not finite contribution.

## Path-Integrated Functional

For path:

```math
\gamma
:
\left[
0,1
\right]
\longrightarrow
\mathcal{X},
```

with endpoints `X_0` and `X`, path attribution is:

```math
\Phi_j^{\gamma}
=
\int_0^1
\frac{
\partial F
\left(
\gamma(t)
\right)
}{
\partial\gamma_j
}
\frac{
\mathrm{d}\gamma_j(t)
}{
\mathrm{d}t
}
\,\mathrm{d}t.
```

Integrated Gradients uses a straight-line path:

```math
\gamma(t)
=
X_0+t(X-X_0).
```

## Perturbation Functional

For component-removal operator `M_j`:

```math
\Phi_j^{\mathrm{remove}}
\left[
F;X
\right]
=
F(X)
-
F
\left(
M_j(X)
\right).
```

This measures effect of the chosen operator, not an intrinsic component value.

## WSI Components Are A Design Choice

The component index can denote pixels, cells, patches, regions, graph nodes,
prototypes, or concepts:

```math
j
\in
\mathcal{I}_{\mathrm{pixel}},
\mathcal{I}_{\mathrm{patch}},
\mathcal{I}_{\mathrm{region}},
\mathcal{I}_{\mathrm{concept}}.
```

Attributions across different component partitions are not directly
comparable.

## Reference Dependence

For two baselines:

```math
X_0
\ne
X_0',
```

generally:

```math
\Phi
\left[
F;X,X_0
\right]
\ne
\Phi
\left[
F;X,X_0'
\right].
```

A blank patch, mean embedding, removed instance, and sampled benign patch each
answer a different reference question.
