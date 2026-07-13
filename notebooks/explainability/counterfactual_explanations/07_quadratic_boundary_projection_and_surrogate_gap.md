# Quadratic Boundary Projection and the Surrogate Gap

## 1. Gaussian Decision Boundary

For two class-conditional latent Gaussians, define discriminants

```math
\delta_c(z)
=-\frac12(z-\mu_c)^{\top}\Sigma_c^{-1}(z-\mu_c)
-\frac12\log|\Sigma_c|+log\pi_c.
```

The boundary between source class `a` and target class `b` satisfies

```math
q_{ab}(z)=\delta_a(z)-\delta_b(z)=0.
```

Expanding gives a quadratic hypersurface:

```math
q_{ab}(z)=z^{\top}Az+b^{\top}z+c.
```

When covariances are equal, `A` vanishes and the boundary is affine. Unequal
covariances produce a genuinely curved boundary.

## 2. Nearest Boundary Projection

GMM-CeFlow seeks

```math
z^{\star}
\in\arg\min_z
\frac12\|z-z_{\mathrm{org}}\|_2^2
\quad\text{subject to}\quad
q_{ab}(z)=0.
```

The Lagrangian is

```math
\mathcal J(z,\eta)
=\frac12\|z-z_{\mathrm{org}}\|_2^2
+\eta q_{ab}(z).
```

Stationarity gives

```math
(I+2\eta A)z
=z_{\mathrm{org}}-\eta b.
```

For admissible `eta`, substitute

```math
z(\eta)
=(I+2\eta A)^{-1}
(z_{\mathrm{org}}-\eta b)
```

into the quadratic constraint and solve the resulting scalar root problem.

## 3. Crossing Tolerance

A point exactly on the boundary is not confidently assigned to the target.
Move a specified tolerance into the target region or require

```math
\delta_b(z_{\mathrm{cf}})-\delta_a(z_{\mathrm{cf}})\ge\kappa.
```

Larger `kappa` improves target margin but increases distance.

## 4. The Surrogate Gap

Let the original black-box WSI model be `F`, while the flow-induced classifier
on HIFs is `g`. Counterfactual validity for the surrogate is

```math
g(u_{\mathrm{cf}})=b.
```

This does not imply

```math
F(X_{\mathrm{cf}})=b
```

because the method does not construct a unique image `X_cf` satisfying the
counterfactual features, and `g` need not reproduce `F` everywhere.

## 5. Required Fidelity Audit

Report held-out agreement

```math
\mathbb P[g(\Phi_{\mathrm{HIF}}(X))=F(X)]
```

and local agreement near each counterfactual path. High global agreement can
hide boundary disagreement exactly where counterfactuals are generated.

