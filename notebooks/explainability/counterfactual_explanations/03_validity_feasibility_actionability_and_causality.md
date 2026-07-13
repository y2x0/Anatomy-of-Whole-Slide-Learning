# Validity, Feasibility, Actionability, and Causality

## 1. Four Nested Requirements

For candidate `x_prime`, define:

```math
\begin{aligned}
\text{validity:}&\quad f(x')\in\mathcal Y_{\star},\\
\text{plausibility:}&\quad x'\in\mathrm{supp}(P_X),\\
\text{feasibility:}&\quad x'\in\mathcal F(x),\\
\text{causal consistency:}&\quad x'=X_{\mathrm{do}(A=a')}(u).
\end{aligned}
```

Validity is a model property. Plausibility is distributional. Feasibility is
case-dependent. Causal consistency requires structural assumptions.

## 2. Plausibility Is Not Feasibility

A point can be common in the cohort yet unreachable from this case. If age and
an acquired measurement are correlated, a plausible record may require making
age younger. Conversely, a rare but physically possible morphology may have low
density under a limited training cohort.

## 3. Immutable and Directional Constraints

Partition coordinates into immutable, monotone, and free sets:

```math
x'_k=x_k
\quad\text{for }k\in\mathcal I,
```

```math
x'_k\ge x_k
\quad\text{for }k\in\mathcal M_{+},
\qquad
x'_k\le x_k
\quad\text{for }k\in\mathcal M_{-}.
```

For histology images, coordinatewise constraints are insufficient because
tissue structures obey coupled geometric and biological relations.

## 4. Manifold Constraints

A generator defines an approximate feasible manifold:

```math
x'=G(z'),
\qquad
z'\in\mathcal Z.
```

Searching in latent space constrains outputs to the generator's range, not the
true tissue manifold. Generator realism is a necessary validation target, not
an assumption that can be granted by architecture.

## 5. Causal Structural Form

For structural equations

```math
X_k=g_k(\mathrm{pa}_k,U_k),
```

an intervention replaces selected equations and recomputes descendants using
the same exogenous state. Ordinary counterfactual explanation optimization
usually modifies coordinates independently and therefore should be called a
model or data-plausible counterfactual, not a patient-level causal claim.

