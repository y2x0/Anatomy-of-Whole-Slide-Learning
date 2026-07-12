# Perturbation As An Interventional Operator

Primary anchors:

- Ribeiro, Singh, Guestrin. "Why Should I Trust You?" KDD 2016.
- Lundberg, Lee. "A Unified Approach to Interpreting Model Predictions."
  NeurIPS 2017.
- Kaczmarzyk et al. "Explainable AI for Computational Pathology Identifies
  Model Limitations and Tissue Biomarkers." 2024.
  https://arxiv.org/abs/2409.03080

## Predictor And Intervention

Let WSI bag be:

```math
X
=
\left\{
x_1,\ldots,x_n
\right\},
```

and scalar target be:

```math
F(X).
```

For selected subset `S`, a perturbation operator is:

```math
\mathcal{T}_S
:
\mathcal{X}
\longrightarrow
\widetilde{\mathcal{X}}.
```

The finite model effect is:

```math
\Delta_F(S;X,\mathcal{T})
=
F(X)
-
F
\left(
\mathcal{T}_S(X)
\right).
```

The result is a property of `F`, `X`, `S`, and the intervention `T`.

## Four Perturbation Questions

Necessity-style removal:

```math
F(X)-F(X\setminus S).
```

Sufficiency-style retention:

```math
F(S)-F(X_0).
```

Replacement:

```math
F(X)
-
F
\left(
X_{\setminus S}
\cup
B_S
\right).
```

Addition:

```math
F(X\cup A)-F(X).
```

These effects are not interchangeable.

## Model Intervention Versus Biological Intervention

Patch removal implements:

```math
\mathrm{do}
\left(
X_{\mathrm{model}}
=
X\setminus S
\right).
```

It does not necessarily implement:

```math
\mathrm{do}
\left(
\text{patient tissue process absent}
\right).
```

The former is a controlled input intervention on the predictor. The latter
requires a biological structural model.

## Perturbation Distribution

If masks are random:

```math
M
\sim
q(M\mid X),
```

the explanation estimates behavior under `q`:

```math
\mathbb{E}_{M\sim q}
\left[
F(X)
-
F
\left(
\mathcal{T}_M(X)
\right)
\right].
```

Changing mask size, spatial coherence, or class conditioning changes the
identified functional.

## WSI Validity

Permutation-invariant MIL accepts variable-size sets, making deletion
syntactically valid. Statistical validity is stronger: the modified set may
have tissue composition or cardinality absent from training.
