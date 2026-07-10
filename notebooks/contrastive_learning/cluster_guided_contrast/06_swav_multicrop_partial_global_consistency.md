# SwAV Multi-Crop And Partial-Global Consistency

SwAV's multi-crop objective asks small views to predict codes computed from
larger views. This creates a directed local-to-global consistency assumption.

Primary anchor:

- SwAV: https://arxiv.org/abs/2006.09882

## View Collection

For one image `x`, sample:

```math
x_{t_1},
x_{t_2}
```

as two standard-resolution views and:

```math
x_{t_3},
\ldots,
x_{t_{V+2}}
```

as `V` additional low-resolution views.

The corresponding normalized representations are:

```math
z_{t_v}
=
\frac{
f_{\theta}
\left(
x_{t_v}
\right)
}{
\left\lVert
f_{\theta}
\left(
x_{t_v}
\right)
\right\rVert_2
},
\qquad
v\in\left\{
1,\ldots,V+2
\right\}.
```

## Codes Only For Global Views

SwAV computes balanced codes for the two standard-resolution views:

```math
q_{t_1},
q_{t_2}.
```

It does not compute Sinkhorn targets for the small crops. The paper reports
that doing so increases computation and degrades transfer performance because
partial views produce weaker assignments.

## Exact Multi-Crop Loss

The paper generalizes swapped prediction to:

```math
\mathcal{L}
\left(
z_{t_1},
\ldots,
z_{t_{V+2}}
\right)
=
\sum_{
i\in\left\{
1,2
\right\}
}
\sum_{v=1}^{V+2}
\mathbf{1}
\left[
v\ne i
\right]
\ell
\left(
z_{t_v},
q_{t_i}
\right).
```

There are:

```math
2
\left(
V+1
\right)
```

directed prediction terms per source image before any averaging convention.

## Directed Supervision Graph

The positive supervision graph contains:

```math
z_{t_2}
\longrightarrow
q_{t_1},
\qquad
z_{t_1}
\longrightarrow
q_{t_2},
```

and for every small view `v\ge 3`:

```math
z_{t_v}
\longrightarrow
q_{t_1},
\qquad
z_{t_v}
\longrightarrow
q_{t_2}.
```

No reverse code target:

```math
q_{t_v}
```

is constructed for a small crop.

## Population Optimum For A Local View

Let `L` denote a small crop and `Q_G` the random balanced code of
a global view from the same source image.

For a predictor:

```math
p_{\theta}
\left(
\cdot
\mid
L
\right),
```

the expected cross-entropy is:

```math
\mathcal{R}
\left(
p_{\theta}
\right)
=
\mathbb{E}
\left[
-
\sum_{k=1}^{K}
Q_G^{(k)}
\log
p_{\theta}^{(k)}
\left(
L
\right)
\right].
```

Conditioning on `L=\ell`:

```math
\mathcal{R}_{\ell}
=
-
\sum_{k=1}^{K}
\mathbb{E}
\left[
Q_G^{(k)}
\mid
L=\ell
\right]
\log
p_{\theta}^{(k)}
\left(
\ell
\right).
```

The unrestricted minimizer is:

```math
p^{\star}
\left(
\cdot
\mid
L=\ell
\right)
=
\mathbb{E}
\left[
Q_G
\mid
L=\ell
\right].
```

A local crop learns the conditional mean global code of the images in which
that local pattern appears.

## What The Local Feature Preserves

If a local tissue pattern always occurs in one global context:

```math
Q_G
=
q_{\star}
\quad
\text{whenever}
\quad
L=\ell,
```

then:

```math
p^{\star}
\left(
\cdot
\mid
\ell
\right)
=
q_{\star}.
```

If the same local pattern appears in several global contexts:

```math
Q_G
\sim
p
\left(
q
\mid
L=\ell
\right),
```

then:

```math
p^{\star}
\left(
\cdot
\mid
\ell
\right)
```

is an average over those contexts. Multi-crop can encode co-occurrence
statistics rather than a unique local tissue identity.

## Pathology Mixture Counterexample

Let a global pathology crop contain tumor `T` and stroma `S`. A
small crop may contain only one:

```math
L_T
\subset
T,
\qquad
L_S
\subset
S.
```

Both small views are trained against the same global codes:

```math
z(L_T)
\longrightarrow
q_G,
\qquad
z(L_S)
\longrightarrow
q_G.
```

If the prototype target is sharply concentrated:

```math
q_G
\approx
e_k,
```

both local states are pressured to predict prototype `k`. The objective
can merge tumor and stroma when their shared global context dominates the
assignment.

## Rare-Focus Omission

Let a diagnostic focus occupy fraction `\rho` of the source image. If
small crops sample area uniformly, the probability a small view misses the
focus is approximately:

```math
1-\rho
```

under a simplified point-sampling model.

A focus-absent small crop is still trained against a global code influenced by
the focus. This can be useful contextual prediction, but it is not guaranteed
to preserve local visual sufficiency.

## Multi-Crop As Conditional Prediction

The objective can be read as:

```math
\text{partial view}
\longrightarrow
\text{balanced code of a broader view}.
```

Its success requires enough shared or predictive information:

```math
I
\left(
L;
Q_G
\right)
>
0.
```

If:

```math
I
\left(
L;
Q_G
\right)
\approx
0,
```

the best local prediction approaches the marginal mean code, which is close to
uniform under exact balancing.

## Computational Effect

Using two global and `V` small views increases encoder work, but the small
views have lower spatial resolution. Balanced assignment is computed for only
two global views:

```math
\mathrm{Sinkhorn\ matrices}
=
2
\quad
\text{per batch view set}.
```

The prediction work scales with:

```math
2
\left(
V+1
\right)
BK
```

prototype-logit comparisons, up to batching and shared computation.

## C/R/G/S Placement

```text
\mathcal{G}:
    two global-code nodes with directed edges from every other view

\mathcal{C}:
    shared encoder and prototype scorer at multiple resolutions

\mathcal{R}:
    balanced codes for global views only

\mathcal{S}:
    every non-self view predicts each global code
```

## Failure Principle

Multi-crop teaches a small region to predict the prototype distribution of its
larger context. In pathology, that may capture useful tissue co-occurrence or
may erase distinctions between local components that inhabit the same global
crop.
