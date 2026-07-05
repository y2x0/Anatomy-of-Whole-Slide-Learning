# Ranking, Calibration, And Time Summaries

Discrete survival models output a curve, but evaluation often compresses the
curve back into one risk score. This creates a tension between likelihood,
ranking, and calibration.

## Horizon Risk

At horizon \(\tau_k\), risk is cumulative incidence:

```math
r_i(\tau_k)
=
F_i(\tau_k)
=
1-S_i(\tau_k)
=
1-\prod_{\ell=1}^{k}(1-h_{i\ell}).
```

The derivative of horizon risk with respect to hazard \(h_{im}\), for \(m\le k\),
is:

```math
\frac{\partial r_i(\tau_k)}{\partial h_{im}}
=
\prod_{\ell\le k,\ell\ne m}(1-h_{i\ell}).
```

Early hazards affect all later horizon risks.

## Expected Event Time

Using interval representatives \(\bar\tau_k\):

```math
\widehat{\mathbb E}[T_i]
=
\sum_{k=1}^{K}\bar\tau_kp_{ik}
+\bar\tau_{>K}p_{i,>K}.
```

Since:

```math
p_{ik}=h_{ik}\prod_{\ell<k}(1-h_{i\ell}),
```

the expected time is a nonlinear function of all hazards.

Different summaries can disagree:

```text
high 1-year risk but low 5-year risk
low early risk but high late risk
same expected time but different curve shape
```

## Pairwise Ranking

A common ranking relation is:

```math
i\prec j
\quad\text{if}\quad
X_i<X_j,\ \delta_i=1.
```

At horizon \(\tau_k\), a pairwise logistic ranking loss can be:

```math
\mathcal L_{\mathrm{rank}}
=
\sum_{i,j}
\mathbf 1[i\prec j]
\log
\left[
1+\exp(-(r_i(\tau_k)-r_j(\tau_k))/\gamma)
\right].
```

or a hinge loss:

```math
\mathcal L_{\mathrm{rank}}
=
\sum_{i,j}
\mathbf 1[i\prec j]
\max(0,1-r_i+r_j).
```

This encourages concordance but does not define a proper likelihood for event
times.

## DeepHit-Style Composite Objective

A generic composite objective is:

```math
\mathcal L
=
\mathcal L_{\mathrm{lik}}
+
\alpha\mathcal L_{\mathrm{rank}}
+
\beta\mathcal L_{\mathrm{cal}}.
```

The design question is which object each term acts on:

```text
likelihood acts on p_k, h_k, or S_k
ranking acts on horizon risk or expected time
calibration acts on predicted probabilities at time horizons
```

If \(\alpha\) is too large, the model may rank well while losing probabilistic
meaning.

## IPCW Brier Score

For horizon \(t\), the binary event indicator is:

```math
Y_i(t)=\mathbf 1[T_i\le t].
```

But \(Y_i(t)\) may be unobserved if subject \(i\) is censored before \(t\).

Let:

```math
G(t)=\Pr(C>t)
```

be the censoring survival function. The IPCW Brier score is:

```math
\operatorname{BS}(t)
=
\frac{1}{n}
\sum_i
\left[
\frac{\mathbf 1[X_i\le t,\delta_i=1]}{\widehat G(X_i)}
(0-\widehat S_i(t))^2
+
\frac{\mathbf 1[X_i>t]}{\widehat G(t)}
(1-\widehat S_i(t))^2
\right].
```

This evaluates calibrated survival probability, not just ranking.

## Calibration At A Horizon

At time \(\tau_k\), group subjects by predicted risk:

```math
\widehat F_i(\tau_k)=1-\widehat S_i(\tau_k).
```

Calibration asks:

```math
\Pr(T\le \tau_k\mid \widehat F(\tau_k)=q)
=
q.
```

Empirically this requires censoring-aware estimates, usually Kaplan-Meier or
IPCW within risk groups.

## WSI Consequence

Suppose the WSI head predicts:

```math
h_i=\sigma(Wz_i+b).
```

Likelihood gradients train each time bin:

```math
\frac{\partial\mathcal L_{\mathrm{lik}}}{\partial z_i}
=
\sum_km_{ik}(h_{ik}-y_{ik})w_k.
```

Ranking at horizon \(K^\star\) trains a compressed risk:

```math
r_i=1-\prod_{k\le K^\star}(1-h_{ik}).
```

So the ranking gradient is:

```math
\frac{\partial r_i}{\partial z_i}
=
\sum_{m\le K^\star}
\left[
\prod_{\ell\le K^\star,\ell\ne m}(1-h_{i\ell})
\right]
h_{im}(1-h_{im})w_m.
```

It mixes time-bin directions according to their effect on the chosen horizon.

## Design Rule

Do not say a discrete survival model outputs "risk" without specifying the risk
functional:

```math
\rho[h_i]
\in
\left\{
F_i(\tau_k),
\widehat{\mathbb E}[T_i],
\sum_kh_{ik},
-S_i(\tau_k),
\operatorname{median}(T_i)
\right\}.
```

Different \(\rho\) define different patient orderings.

## Dense Summary

```math
\begin{aligned}
r_i(\tau_k)&=1-\prod_{\ell\le k}(1-h_{i\ell}),\\
\frac{\partial r_i(\tau_k)}{\partial h_{im}}
&=
\prod_{\ell\le k,\ell\ne m}(1-h_{i\ell}),\\
\mathcal L
&=
\mathcal L_{\mathrm{lik}}
+\alpha\mathcal L_{\mathrm{rank}}
+\beta\mathcal L_{\mathrm{cal}}.
\end{aligned}
```

Discrete survival outputs a curve. Any scalar risk score is a chosen functional
of that curve.
