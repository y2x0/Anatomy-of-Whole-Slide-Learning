# Explanation Nonidentifiability

Primary anchors:

- Jain, Wallace. "Attention is not Explanation." NAACL 2019.
  https://arxiv.org/abs/1902.10186
- Wiegreffe, Pinter. "Attention is not not Explanation." EMNLP 2019.
  https://arxiv.org/abs/1908.04626

## Same Prediction, Different Explanation

Let attention-pooled representation be:

```math
z
=
\sum_{j=1}^{n}
\alpha_j h_j,
\qquad
\alpha
\in
\Delta^{n-1}.
```

If there exists nonzero `delta` such that:

```math
\sum_j
\delta_j h_j
=
0,
\qquad
\sum_j\delta_j=0,
```

then:

```math
\alpha'
=
\alpha+\delta
```

produces the same pooled vector whenever `alpha'` remains in the simplex.

Thus:

```math
\alpha\ne\alpha',
\qquad
z(\alpha)=z(\alpha').
```

Attention weights are not identifiable from the prediction alone.

## Null Space Dimension

Stack patch features as columns:

```math
H
=
\left[
h_1,\ldots,h_n
\right]
\in
\mathbb{R}^{d\times n}.
```

Prediction-preserving perturbations satisfy:

```math
H\delta=0,
\qquad
\mathbf{1}^{\top}\delta=0.
```

When `n` greatly exceeds `d`, as in WSI bags, this constrained null space can
be high dimensional.

## Same Function, Different Internals

Suppose invertible matrix `A` transforms hidden features:

```math
h_j'
=
Ah_j,
```

and downstream weights transform inversely. The predictor can remain exactly
unchanged while internal directions, prototypes, or activation maps change.

## Attribution Underdetermination

Completeness imposes one scalar constraint:

```math
\sum_j\phi_j
=
F(X)-F(X_0).
```

For `m` components, infinitely many vectors satisfy this when `m>1`. Additional
axioms and reference choices choose one attribution from an underdetermined
set.

## Agreement Is Not Validation

If two methods agree:

```math
\mathcal{E}_1(F,X)
\approx
\mathcal{E}_2(F,X),
```

they may share the same baseline, gradient saturation, or model shortcut.
Agreement does not identify a ground-truth explanation.

## Proper Claim

An attention map can be described as the model's learned readout weights. It
cannot be promoted to a unique causal importance map without additional tests
and assumptions.
