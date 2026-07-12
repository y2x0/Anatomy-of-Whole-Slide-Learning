# The Explanation Target Ladder

Primary anchors:

- Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML
  2017.
- Ribeiro, Singh, Guestrin. "Why Should I Trust You?" KDD 2016.
- Wachter, Mittelstadt, Russell. "Counterfactual Explanations without Opening
  the Black Box." 2017.

## Predictor And Observed Object

Let a WSI be represented by patches and optional coordinates:

```math
X
=
\left\{
\left(
x_j,c_j
\right)
\right\}_{j=1}^{n}.
```

A predictor is:

```math
F
:
\mathcal{X}
\longrightarrow
\mathbb{R}^{C}.
```

For class `c`, write its score as:

```math
F_c(X).
```

An explanation method is another map:

```math
\mathcal{E}
:
\left(
F,X,c,\mathcal{B}
\right)
\longrightarrow
E,
```

where `B` contains baselines, perturbation laws, concept sets, or reference
data. The explanation is not a property of `X` alone.

## Five Distinct Targets

Score explanation:

```math
E
\approx
\text{decomposition of }F_c(X).
```

Decision explanation:

```math
E
\approx
\text{reason }
\arg\max_k F_k(X)=c.
```

Contrastive explanation:

```math
E
\approx
\text{reason }F_c(X)-F_{c'}(X)>0.
```

Data explanation:

```math
E
\approx
\text{association between observed morphology and outcome}.
```

Causal explanation:

```math
E
\approx
\text{effect of an intervention on morphology or tissue state}.
```

Only the first three are directly determined by a fixed predictive model.

## Score And Probability Are Different Targets

Let logits be `f_k(X)` and probability be:

```math
p_c(X)
=
\frac{
\exp
\left(
f_c(X)
\right)
}{
\sum_k
\exp
\left(
f_k(X)
\right)
}.
```

Then:

```math
\nabla_X p_c
=
p_c
\left[
\nabla_X f_c
-
\sum_k
p_k\nabla_X f_k
\right].
```

Attributing `f_c` explains one logit. Attributing `p_c` explains that logit
relative to all candidate logits. The heatmaps need not agree.

## Prediction And Correctness

An explanation can be perfectly faithful to an incorrect model:

```math
\mathcal{E}(F,X)
\text{ faithful to }F
\not\Rightarrow
F(X)
\text{ clinically correct}.
```

Similarly:

```math
F(X)
\text{ correct}
\not\Rightarrow
\mathcal{E}(F,X)
\text{ faithful}.
```

Predictive validity and explanatory validity are separate axes.

## WSI Target Ambiguity

A patch heatmap can target:

```math
\text{patch contribution to slide score},
```

```math
\text{patch selection by attention},
```

```math
\text{patch similarity to a prototype},
```

or:

```math
\text{predicted lesion probability}.
```

Calling all four tumor localization is an unsupported target substitution.
