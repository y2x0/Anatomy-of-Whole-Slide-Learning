# Counterfactual Objects and Optimization

## 1. Generic Definition

For model `f`, observed input `x`, and desired output set `Y_star`, a
model counterfactual solves

```math
x^{\star}
\in
\arg\min_{x'\in\mathcal F(x)}
d_x(x,x')
\quad
\text{subject to}
\quad
f(x')\in\mathcal Y_{\star}.
```

The explanation is inseparable from four choices:

```math
(f,\mathcal Y_{\star},d_x,\mathcal F(x)).
```

Here `F(x)` is the feasible set. Changing any member can change the nearest
counterfactual.

## 2. Three Counterfactual State Spaces

For WSI learning, the decision may be changed in:

```math
\begin{aligned}
\text{bag space:}&\quad B'=B\setminus A\text{ or }B\cup A,\\
\text{feature space:}&\quad u'=u+\delta,\\
\text{image space:}&\quad x'=G(z').
\end{aligned}
```

Bag edits explain dependence on observed instances. Feature edits explain a
surrogate or feature-based decision rule. Image synthesis proposes altered
morphology under a generator. None automatically implies a physically possible
change to the same patient specimen.

## 3. Validity and Minimality

Define target validity and cost:

```math
V(x')=\mathbf 1\{f(x')\in\mathcal Y_{\star}\},
\qquad
C(x,x')=d_x(x,x').
```

A valid solution need not be minimal, and a minimal numerical change need not
be meaningful. If the feasible set is nonconvex, local optimization can return
different solutions from different initializations.

## 4. Explanation Versus Recourse

A counterfactual explanation describes a model dependency. Recourse additionally
requires an actionable intervention available to the affected agent. In
pathology, changing nuclear morphology is not patient recourse; it is a virtual
experiment on the model's image representation.

## 5. Explanation Versus Causal Counterfactual

Model validity states

```math
f(x')\ne f(x).
```

A causal counterfactual requires a structural model that generates downstream
variables consistently after an intervention. Optimization against `f` alone
does not provide that model.

