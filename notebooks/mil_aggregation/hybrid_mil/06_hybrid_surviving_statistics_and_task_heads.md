# Hybrid Surviving Statistics And Task Heads

## 1. Context does not determine the task representation

Let a hybrid context operator produce contextual states T_i. The same T_i can
feed distinct task heads:

```math
z_i^{(r)}
=
\mathcal R_r(T_i),
\qquad
\widehat y_i^{(r)}
=
\mathcal H_r(z_i^{(r)}).
```

Changing only R changes which statistic the task can observe. Changing C changes
the sufficient information available to every R.

## 2. Classification heads

For binary classification with a linear head,

```math
\ell_i
=
w^{\mathsf T}z_i+b,
\qquad
\widehat p_i=\sigma(\ell_i).
```

If z_i is a mean of contextual states, the head sees a first moment. If z_i is
max pooled, it sees coordinatewise order statistics. If z_i is a prototype
occupancy vector, it sees soft counts in learned regions of feature space.

A graph or transformer context can make these statistics relational even when the
final readout is a mean.

## 3. Cox survival heads

For a scalar risk representation,

```math
\eta_i
=
w^{\mathsf T}z_i+b,
\qquad
h(t\mid z_i)
=
h_0(t)\exp(\eta_i).
```

The partial log-likelihood is

```math
\mathcal L_{\mathrm{Cox}}
=
-\sum_{i:\delta_i=1}
\left[
\eta_i
-
\log
\sum_{j\in\mathcal R(t_i)}
\exp(\eta_j)
\right].
```

A hybrid context can improve the covariate z_i, but the Cox head still uses one
patient-specific risk score. It cannot represent time-varying crossings unless
the architecture produces a richer time-indexed output or the proportional
hazards restriction is relaxed.

## 4. Discrete hazard heads

At horizons tau_1 through tau_K, a vector head gives

```math
\eta_{ik}
=
w_k^{\mathsf T}z_i+b_k,
\qquad
h_{ik}=\sigma(\eta_{ik}).
```

The survival probability at the end of interval k is

```math
\widehat S_i(\tau_k)
=
\prod_{r=1}^{k}(1-h_{ir}).
```

A hybrid representation can be judged by whether it preserves information useful
at multiple horizons. A single scalar slide mean may be adequate for a Cox head
but insufficient for a non-proportional hazard head when time-specific morphology
matters.

## 5. Statistic composition

Suppose contextualization is linear T=AH and readout is a linear weight vector
a:

```math
z
=
a^{\mathsf T}AH
=
(A^{\mathsf T}a)^{\mathsf T}H.
```

The effective fine-level weights are A^{\mathsf T}a. For a graph propagation
matrix, the task head is weighting graph-diffused evidence, not raw patches.

If R is nonlinear attention, then

```math
z
=
\sum_j
\frac{\exp q((AH)_j)}
{\sum_k\exp q((AH)_k)}
(AH)_j,
```

and the effective weights depend on all instances. In this regime, no fixed
linear surviving statistic exists globally.

## 6. Multi-head task supervision

A shared hybrid representation can support classification and survival heads:

```math
\widehat y_i=\mathcal H_{\mathrm{cls}}(z_i),
\qquad
\eta_i=\mathcal H_{\mathrm{cox}}(z_i),
```

```math
\mathcal L
=
\mathcal L_{\mathrm{cls}}
+
\lambda_{\mathrm{surv}}\mathcal L_{\mathrm{Cox}}.
```

The shared context is trained by both gradients. A feature that is useful for
classification may dominate z even if it is weakly related to survival, so the
surviving statistic must be evaluated task by task.

## 7. Reporting rule

For a hybrid model, report the tuple

```math
\left(
\mathcal C,
\mathcal R,
G,
\mathcal H,
\mathcal L
\right),
```

not only the final head name. At minimum state:

`text
- whether context sees coordinates, adjacency, hierarchy, or order;
- whether the readout is sum, mean, max, attention, prototype, or [CLS];
- whether the task head is scalar risk, hazards, classification logits, or
  retrieval distances;
- which intermediate losses modify the representation;
- which statistic is expected to survive each boundary.
`
`
