# Faithfulness, Plausibility, and Causality

## 1. Three Properties

For explanation `E` and model output `F`, define:

```math
\text{faithfulness:}
\quad E\text{ tracks a specified change in }F,
```

```math
\text{plausibility:}
\quad E\text{ is coherent to an observer or lies near }P_X,
```

```math
\text{causal validity:}
\quad E\text{ corresponds to a valid structural intervention}.
```

These properties are logically independent. A visually plausible attention map
can be unfaithful; a faithful adversarial perturbation can be implausible; a
plausible image edit can lack causal interpretation.

## 2. Faithfulness Functionals

For scalar target `F_c`, deletion faithfulness can use

```math
\rho_{\mathrm{del}}
=\mathrm{Corr}
\left(E_j,
F_c(X)-F_c(X^{(-j)})\right).
```

For survival curves, use a horizon measure:

```math
\rho_S
=\mathrm{Corr}
\left(E_j,
\int_0^\tau
|S(t)-S^{(-j)}(t)|\,d\mu(t)
\right).
```

The intervention semantics must be specified before interpreting the number.

## 3. Causal Caution

```math
\text{model intervention}
\nequivalence
\text{biological intervention}.
```

Graph node deletion, concept zeroing, and diffusion feature shifts are causal
with respect to a computational graph only under their defined model semantics.

## Intervention Semantics

Let I_a be an intervention on object a and let q be the stated target. Model
faithfulness concerns

```math
\Delta_q(a)
=
q(F(X))-q(F(\mathsf I_a(X))).
```

Biological causal validity concerns a different quantity, for example

```math
\Delta_Y(a)
=
\mathbb E[Y\mid\mathrm{do}(A=a)]
-
\mathbb E[Y\mid\mathrm{do}(A=a_0)].
```

There is no implication from a large model delta to a large biological effect.
The implication requires an intervention map from computational object to a
valid structural variable, plus assumptions that support transport from observed
slides to the intervention distribution.

For a ranked explanation E, deletion faithfulness can be written as a family of
correlations over interventions:

```math
\rho_{\mathrm{del}}
=
\mathrm{Corr}_{a\sim\mathcal Q}
\left(
E_a,
\Delta_q(a)
\right).
```

Changing the intervention distribution Q changes the claim being measured. A
patch deletion test, a feasible stain-preserving edit, and a graph rewire are not
interchangeable evaluations.
