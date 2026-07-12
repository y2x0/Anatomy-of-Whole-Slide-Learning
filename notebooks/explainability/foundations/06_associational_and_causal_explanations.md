# Associational And Causal Explanations

Primary anchor:

- Wachter, Mittelstadt, Russell. "Counterfactual Explanations without Opening
  the Black Box." 2017.

## Predictive Association

A model estimates some functional of:

```math
p(Y\mid X).
```

An attribution to morphology component `X_j` explains dependence of the model
score on observed input. It does not automatically estimate:

```math
p
\left(
Y
\mid
\mathrm{do}(X_j=x)
\right).
```

## Structural Variables

Let:

```math
U
=
\text{patient and acquisition factors},
```

```math
M
=
\text{biological morphology},
```

```math
X
=
\text{digitized appearance},
```

```math
Y
=
\text{clinical target}.
```

A plausible structural graph contains:

```math
U
\longrightarrow
M
\longrightarrow
X,
```

```math
U
\longrightarrow
X,
\qquad
M
\longrightarrow
Y.
```

Stain or scanner factors can predict `Y` through cohort association without
being causal disease mechanisms.

## Three Counterfactual Objects

Model counterfactual:

```math
F(X')
\ne
F(X).
```

Data-plausible counterfactual:

```math
X'
\in
\mathrm{supp}
\left(
p_X
\right).
```

Causal counterfactual:

```math
X'
=
X
\left(
\mathrm{do}(M_j=m')
\right)
```

under a structural causal model. The first does not imply the second or third.

## Confounding Counterexample

Suppose institution `S` affects stain `X_s` and label prevalence:

```math
S
\longrightarrow
X_s,
\qquad
S
\longrightarrow
Y.
```

Then:

```math
I(X_s;Y)>0
```

even if:

```math
Y
\perp
X_s
\mid
S.
```

A faithful saliency map may highlight stain because the predictor uses it. That
is evidence of a model shortcut, not evidence that stain causes disease.

## Intervention Feasibility

Changing one histologic concept while holding all else fixed may violate
biological dependencies:

```math
p
\left(
M_j=m',
M_{-j}=m_{-j}
\right)
=
0.
```

Counterfactual optimization must specify feasibility and causal consistency,
not only image realism.

## Proper Claim

Most post-hoc WSI explanations are associational explanations of model
behavior. Causal language requires an intervention model and assumptions beyond
the trained predictor.
