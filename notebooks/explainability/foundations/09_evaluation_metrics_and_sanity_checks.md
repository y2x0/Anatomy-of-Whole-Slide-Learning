# Evaluation Metrics And Sanity Checks

## Deletion Curve

Order components by explanation score and remove the first `k`:

```math
X_{-I_k}
=
\mathcal{T}_{\mathrm{delete}}
\left(
X,I_k
\right).
```

Deletion response is:

```math
D(k)
=
F(X)
-
F
\left(
X_{-I_k}
\right).
```

A steep curve suggests selected components support the score under the chosen
deletion operator.

## Insertion Curve

Starting from baseline `X_0`:

```math
X_{0,+I_k}
=
\mathcal{T}_{\mathrm{insert}}
\left(
X_0,X,I_k
\right),
```

```math
I(k)
=
F
\left(
X_{0,+I_k}
\right)
-
F(X_0).
```

Deletion and insertion need not agree because their paths traverse different
off-manifold states.

## Localization Metrics

For annotation mask `G` and thresholded explanation `E_tau`:

```math
\mathrm{IoU}
=
\frac{|E_{\tau}\cap G|}{|E_{\tau}\cup G|}.
```

Localization evaluates overlap with annotation. It does not test whether the
highlighted area actually drives the model.

## Pointing Game

```math
\mathrm{Hit}
=
\mathbb{1}
\left\{
\arg\max_j E_j
\in
G
\right\}.
```

This tests only the maximum location and ignores explanation extent, false
positive regions, and score faithfulness.

## Model Randomization

Let `F^(l)` denote the model after randomizing layers through depth `l`.
A parameter-sensitive explanation should change:

```math
d_E
\left(
\mathcal{E}(F,X),
\mathcal{E}(F^{(l)},X)
\right)
\uparrow
```

as learned structure is destroyed. Persistence under complete randomization
suggests the method primarily reflects input structure or architecture priors.

## Label Randomization

Train model `F_perm` on permuted labels. Explanations should not retain the
same task-specific patterns:

```math
\mathcal{E}(F_{\mathrm{perm}},X)
\not\approx
\mathcal{E}(F,X).
```

## Human Agreement

Pathologist agreement tests plausibility:

```math
\mathrm{Plausibility}
=
\mathrm{sim}
\left(
E,G_{\mathrm{human}}
\right).
```

It is not a substitute for faithfulness because a model can use a shortcut
while an explanation method produces a plausible map.

## Evaluation Matrix

At minimum, explanations should be evaluated on:

```math
\text{model faithfulness},
\quad
\text{stability},
\quad
\text{localization or concept validity},
\quad
\text{randomization sensitivity}.
```
