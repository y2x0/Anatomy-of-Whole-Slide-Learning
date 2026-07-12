# Integrated Gradients Baselines, Paths, And WSI Bags

Primary anchor:

- Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML
  2017.

## Baseline Is The Missingness Claim

Integrated Gradients allocates:

```math
F(x)-F(x').
```

It does not allocate `F(x)` without qualification. Different baselines answer
different questions.

For WSI patch embedding `h_j`, possible baselines include:

```math
h_j'=0,
```

```math
h_j'
=
\mathbb{E}
\left[
H_{\mathrm{benign}}
\right],
```

or a sampled tissue embedding. Zero need not correspond to absent tissue.

## Straight Paths Can Leave Support

```math
h_j(\alpha)
=
h_j'
+
\alpha
\left(
h_j-h_j'
\right).
```

Intermediate embeddings may not be realizable by any histology patch:

```math
h_j(\alpha)
\notin
\mathrm{supp}
\left(
p_H
\right).
```

The model is queried along an off-manifold path.

## Path Dependence

For a general path `gamma`:

```math
\mathrm{PIG}_j^{\gamma}
=
\int_0^1
\frac{
\partial F
\left(
\gamma(\alpha)
\right)
}{
\partial\gamma_j
}
\frac{
\mathrm{d}\gamma_j
}{
\mathrm{d}\alpha}
\,\mathrm{d}\alpha.
```

The sum is path independent for a conservative gradient field, but the
coordinatewise allocation can vary by path because interaction credit is
distributed differently.

## Bag Cardinality Problem

IG requires fixed-dimensional endpoints. For variable-size WSI bags, common
embedding-space construction keeps `n` slots and interpolates each patch:

```math
H(\alpha)
=
H'
+
\alpha(H-H').
```

This is not equivalent to inserting patches one by one. Attention normalization
sees every baseline slot throughout the path.

## Patch-Level IG

Featurewise attributions can be summed:

```math
E_j^{\mathrm{IG}}
=
\sum_{d=1}^{D}
\mathrm{IG}_{jd}
\left(
H;H'
\right).
```

Completeness gives:

```math
\sum_j
E_j^{\mathrm{IG}}
=
F(H)-F(H').
```

The map is signed and context-sensitive through the full model path.

## Expected Baselines

For reference distribution `p_0`:

```math
\overline E_j
=
\mathbb{E}_{H'\sim p_0}
\left[
E_j^{\mathrm{IG}}(H;H')
\right].
```

This reduces dependence on one baseline while changing the target to expected
score difference against the reference population.
