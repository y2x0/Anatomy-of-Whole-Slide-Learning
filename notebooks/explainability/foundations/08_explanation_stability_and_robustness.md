# Explanation Stability And Robustness

## Input Stability

For perturbation metric `d_X` and explanation metric `d_E`, local Lipschitz
stability asks whether:

```math
d_E
\left(
\mathcal{E}(F,X),
\mathcal{E}(F,X')
\right)
\le
L_E
d_X(X,X')
```

for nearby inputs.

## Prediction-Conditioned Stability

The strongest concern is explanation change under prediction preservation:

```math
F(X)
\approx
F(X'),
```

but:

```math
d_E
\left(
\mathcal{E}(F,X),
\mathcal{E}(F,X')
\right)
\gg
0.
```

Such instability means the explanation is not a robust account of the stable
decision.

## Parameter Stability

For independently trained models `F_theta` and `F_theta'` with similar
predictions:

```math
\mathbb{E}_X
\left[
\left|
F_{\theta}(X)
-
F_{\theta'}(X)
\right|
\right]
\le
\varepsilon,
```

explanation reproducibility measures:

```math
\mathbb{E}_X
\left[
d_E
\left(
\mathcal{E}(F_{\theta},X),
\mathcal{E}(F_{\theta'},X)
\right)
\right].
```

Different seeds can recover distinct predictive shortcuts with similar test
metrics.

## Rank Stability

For patch scores `s_j`, define top-k set:

```math
I_k(X)
=
\mathrm{TopK}
\left(
\left\{
s_j(X)
\right\},k
\right).
```

Jaccard stability is:

```math
J_k(X,X')
=
\frac{
|I_k(X)\cap I_k(X')|
}{
|I_k(X)\cup I_k(X')|
}.
```

High correlation of dense heatmaps can coexist with unstable top-ranked biopsy
regions.

## Boundary Gap

If sorted scores satisfy:

```math
g_k
=
s_{(k)}-s_{(k+1)},
```

top-k support is stable under score perturbations smaller than half the gap in
supremum norm:

```math
\|\Delta s\|_{\infty}
<
\frac{g_k}{2}.
```

Small gaps make localization inherently brittle.

## Stain And Resolution Stability

For nuisance transformation `T` that should preserve diagnosis:

```math
F(TX)
\approx
F(X),
```

one can separately test:

```math
\mathcal{E}(F,TX)
\approx
T
\mathcal{E}(F,X).
```

This is explanation equivariance, stronger than prediction invariance.

## Stability Is Not Faithfulness

A constant heatmap is perfectly stable and uninformative. Stability must be
reported alongside perturbation faithfulness and task validity.
