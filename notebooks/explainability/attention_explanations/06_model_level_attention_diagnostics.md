# Model-Level Attention Diagnostics

Primary anchor:

- Wiegreffe, Pinter. "Attention is not not Explanation." EMNLP 2019.
  https://arxiv.org/abs/1908.04626

## Distribution Is Not A Free Primitive

In a trained model:

```math
\alpha(X)
=
\mathrm{softmax}
\left(
s_{\theta}(X)
\right).
```

An arbitrary replacement `alpha'` may not be realizable by the learned scoring
mechanism on any input:

```math
\alpha'
\notin
\left\{
\alpha_{\theta}(X):
X\in\mathcal{X}
\right\}.
```

Counterfactual weight replacement and mechanism evaluation answer different
questions.

## Uniform-Attention Baseline

Replace learned attention with:

```math
\alpha_j^{\mathrm{uniform}}
=
\frac{1}{n}.
```

Compare task risks:

```math
\Delta R
=
R
\left(
F_{\mathrm{uniform}}
\right)
-
R
\left(
F_{\mathrm{learned}}
\right).
```

Positive `Delta R` indicates the attention mechanism contributes predictive
value. It does not prove each individual weight is a faithful local credit.

## Frozen-Random Baseline

Let attention scoring parameters be randomly initialized and frozen:

```math
\theta_a
\sim
p_{\mathrm{init}},
\qquad
\nabla\theta_a=0.
```

Training the remaining model tests whether learned attention structure matters
beyond a fixed random pooling pattern.

## Seed Variance

Train `M` models with seeds `1,...,M`. Prediction agreement and attention
agreement are distinct:

```math
A_F
=
\frac{2}{M(M-1)}
\sum_{r<s}
\mathrm{sim}
\left(
F_r(X),F_s(X)
\right),
```

```math
A_{\alpha}
=
\frac{2}{M(M-1)}
\sum_{r<s}
\mathrm{sim}
\left(
\alpha_r(X),\alpha_s(X)
\right).
```

High prediction agreement with low attention agreement reveals explanatory
underdetermination across training runs.

## Mechanism-Constrained Adversary

Instead of directly replacing weights, train adversarial attention parameters:

```math
\min_{\theta_a'}
\mathcal{L}_{\mathrm{task}}
\left(
\theta_a'
\right)
-
\lambda
D
\left(
\alpha_{\theta_a'},
\alpha_{\theta_a}
\right).
```

This asks whether a realizable learned mechanism can produce different
attention while preserving task performance.

## Diagnostic Matrix

Weight permutation tests local necessity. Uniform and random baselines test
mechanism utility. Seed tests assess reproducibility. Adversarial retraining
tests model-level nonuniqueness. No single diagnostic settles every definition
of explanation.
