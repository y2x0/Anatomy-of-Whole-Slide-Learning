# Loss Family Map

A survival model has two separate design choices:

```text
output object
training objective
```

The output object might be:

```math
\eta_i,
\qquad
h_{ik},
\qquad
\lambda_i(t),
\qquad
S_i(t),
\qquad
p_{ik},
\qquad
p_{ikc}.
```

The loss tells the model which property of that object matters.

## Cox Partial Likelihood

Output:

```math
\eta_i=f_\theta(z_i).
```

Loss:

```math
\mathcal{L}_{\operatorname{Cox}}
=
-
\sum_{i:\delta_i=1}
\left[
\eta_i
-
\log\sum_{j\in R_i}\exp(\eta_j)
\right].
```

Meaning:

```text
rank event patients above their risk sets
```

## Continuous Hazard Likelihood

Output:

```math
\lambda_i(t)=\lambda_\theta(t,z_i).
```

Loss:

```math
\mathcal{L}_{i}
=
-\delta_i\log\lambda_i(X_i)
+\int_0^{X_i}\lambda_i(u)\,du.
```

Meaning:

```text
high hazard at event times; low accumulated hazard before observed times
```

## Discrete Hazard Likelihood

Output:

```math
h_{ik}=\Pr(T_i\in I_k\mid T_i>\tau_{k-1},z_i).
```

Loss:

```math
\mathcal{L}_{i}
=
-
\sum_km_{ik}
[y_{ik}\log h_{ik}+(1-y_{ik})\log(1-h_{ik})].
```

Meaning:

```text
masked likelihood over observed at-risk intervals
```

## PMF Likelihood

Output:

```math
p_{ik}=\Pr(T_i\in I_k\mid z_i).
```

Loss for event interval $r$:

```math
-\log p_{ir}.
```

Loss for censoring at $r$:

```math
-\log\sum_{k>r}p_{ik}.
```

Meaning:

```text
learn the discrete event-time distribution directly
```

## Ranking Loss

Output:

```math
r_i=\rho(\text{risk object}_i).
```

Pairwise loss:

```math
\mathcal{L}_{\operatorname{rank}}
=
\sum_{i,j}
\mathbf{1}[X_i<X_j,\delta_i=1]
\phi(r_j-r_i).
```

Meaning:

```text
event patients should have higher risk than later-observed patients
```

## IPCW Horizon Loss

At horizon $t$, target:

```math
Y_i(t)=\mathbf{1}[T_i\le t].
```

Observed only when:

```text
event before t, or still observed past t
```

IPCW loss:

```math
\mathcal{L}_{t}
=
\sum_iw_i(t)\ell(Y_i(t),\widehat{F}_i(t)).
```

Meaning:

```text
censoring-weighted supervised learning at fixed horizon
```

## Dense Summary

```text
Cox loss trains ordering.
Hazard likelihood trains a time process.
Discrete likelihood trains an at-risk sequence.
PMF likelihood trains an event-time distribution.
Ranking loss trains concordance-like behavior.
IPCW loss trains horizon-specific probability.
```

Losses are not interchangeable. They define different statistical problems.
