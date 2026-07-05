# Multi-Objective Losses And Regularization

Modern neural survival models often combine likelihood, ranking, calibration,
and representation regularizers.

The generic objective is:

```math
\mathcal{L}
=
\lambda_{\operatorname{lik}}\mathcal{L}_{\operatorname{lik}}
+\lambda_{\operatorname{rank}}\mathcal{L}_{\operatorname{rank}}
+\lambda_{\operatorname{cal}}\mathcal{L}_{\operatorname{cal}}
+\lambda_{\operatorname{reg}}\mathcal{L}_{\operatorname{reg}}.
```

This is not just engineering. Each term optimizes a different property.

## Likelihood Plus Ranking

For a discrete PMF model:

```math
p_{ik}=\Pr(T_i\in I_k\mid z_i).
```

Likelihood:

```math
\mathcal{L}_{\operatorname{lik}}
=
-
\sum_i
\left[
\delta_i\log p_{i\kappa_i}
+
(1-\delta_i)\log S_i(X_i)
\right].
```

Ranking:

```math
\mathcal{L}_{\operatorname{rank}}
=
\sum_{i,j}
\mathbf{1}[X_i<X_j,\delta_i=1]
\phi(r_j-r_i).
```

If $\lambda_{\operatorname{rank}}$ is too large, the model can sacrifice
calibration for ordering.

## Smoothness Across Time

For discrete hazards:

```math
h_i=(h_{i1},\ldots,h_{iK}).
```

A smoothness penalty:

```math
\mathcal{L}_{\operatorname{smooth}}
=
\sum_i\sum_{k=2}^{K-1}
(g_{i,k+1}-2g_{ik}+g_{i,k-1})^2
```

where $g_{ik}=\operatorname{logit}(h_{ik})$, penalizes jagged hazards.

This is useful when event bins are sparse, but it imposes an assumption that risk
changes smoothly over time.

## Monotonicity Penalties

Survival must be monotone:

```math
S_i(\tau_{k+1})\le S_i(\tau_k).
```

Hazard and PMF parameterizations enforce this automatically. Direct survival
outputs may need:

```math
\mathcal{L}_{\operatorname{mono}}
=
\sum_i\sum_k
\max(0,S_i(\tau_{k+1})-S_i(\tau_k))^2.
```

Prefer parameterizations that make invalid curves impossible.

## Attention Regularization

For WSI attention:

```math
a_i\in\Delta^{n_i-1}.
```

Entropy penalty:

```math
\mathcal{L}_{\operatorname{ent}}
=
\sum_i\sum_j a_{ij}\log a_{ij}.
```

Depending on sign, this encourages either diffuse or sparse attention.

Diversity across cause/time heads:

```math
\mathcal{L}_{\operatorname{div}}
=
\sum_i\sum_{c<c'}
(a_{ic}^\top a_{ic'})^2.
```

This discourages different event heads from selecting identical patches.

## Graph Regularization

For graph survival, one can penalize nonsmooth node risk:

```math
\mathcal{L}_{\operatorname{graph}}
=
\sum_{(u,v)\in E_i}
A_{uv}
\|r_{iu}-r_{iv}\|^2.
```

This imposes spatial smoothness. It may be wrong when sharp boundaries are
prognostic.

## Foundation-Feature Regularization

If patch features come from a frozen foundation model, the survival model may
learn only:

```math
\mathcal{R},\mathcal{H}.
```

If the foundation model is fine-tuned, one may regularize representation drift:

```math
\mathcal{L}_{\operatorname{drift}}
=
\sum_{i,j}
\|E_\theta(x_{ij})-E_0(x_{ij})\|^2.
```

This trades task adaptation against overfitting small survival cohorts.

## Dense Summary

Multi-objective survival losses should be read as:

```text
likelihood: probability model
ranking: discrimination
calibration: horizon probability correctness
smoothness: time regularity
attention/graph regularization: representation bias
```

The combined loss is a design statement. It should not be presented as a single
generic "survival loss."
