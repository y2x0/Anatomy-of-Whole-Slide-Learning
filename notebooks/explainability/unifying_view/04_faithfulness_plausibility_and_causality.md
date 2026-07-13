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

