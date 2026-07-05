# Score, Hessian, And Baseline Recovery

This note writes the Cox family as an optimization problem over risk-set moments.

Let:

```math
\eta_i=\beta^\top z_i
```

for the linear Cox case. Define risk-set weighted moments:

```math
S^{(r)}(t;\beta)
=
\sum_{j\in R(t)}
z_j^{\otimes r}\exp(\beta^\top z_j),
\qquad r=0,1,2.
```

Here:

```math
z^{\otimes0}=1,
\qquad
z^{\otimes1}=z,
\qquad
z^{\otimes2}=zz^\top.
```

Define the risk-set mean:

```math
\bar{z}(t;\beta)
=
\frac{S^{(1)}(t;\beta)}{S^{(0)}(t;\beta)}.
```

and risk-set covariance:

```math
V(t;\beta)
=
\frac{S^{(2)}(t;\beta)}{S^{(0)}(t;\beta)}
-
\bar{z}(t;\beta)\bar{z}(t;\beta)^\top.
```

## Score

The partial log likelihood is:

```math
\ell(\beta)
=
\sum_{i:\delta_i=1}
\left[
\beta^\top z_i
-
\log
\sum_{j\in R_i}\exp(\beta^\top z_j)
\right].
```

Its gradient is:

```math
U(\beta)
=
\frac{\partial \ell}{\partial \beta}
=
\sum_{i:\delta_i=1}
\left[
z_i-\bar{z}(X_i;\beta)
\right].
```

Thus the score equation is:

```math
\sum_{i:\delta_i=1}z_i
=
\sum_{i:\delta_i=1}\bar{z}(X_i;\beta).
```

The event features must match the model-weighted risk-set features.

## Hessian

The Hessian is:

```math
\frac{\partial^2\ell}{\partial\beta\partial\beta^\top}
=
-
\sum_{i:\delta_i=1}
V(X_i;\beta).
```

Since each \(V(X_i;\beta)\) is positive semidefinite:

```math
\ell(\beta)
```

is concave in the linear Cox model.

The negative log partial likelihood has Hessian:

```math
\nabla^2[-\ell(\beta)]
=
\sum_{i:\delta_i=1}V(X_i;\beta).
```

## Neural Cox Score

For \(\eta_i=f_\theta(z_i)\), define:

```math
p_{ij}
=
\frac{\exp(\eta_j)}
{\sum_{k\in R_i}\exp(\eta_k)}
\mathbf{1}[j\in R_i].
```

The event-level loss is:

```math
\mathcal{L}_i
=
-\eta_i+\log\sum_{j\in R_i}\exp(\eta_j).
```

The derivative with respect to each risk score is:

```math
\frac{\partial\mathcal{L}_i}{\partial\eta_j}
=
p_{ij}-\mathbf{1}[j=i].
```

Then:

```math
\nabla_\theta\mathcal{L}_i
=
\sum_{j\in R_i}
(p_{ij}-\mathbf{1}[j=i])
\nabla_\theta f_\theta(z_j).
```

The gradient couples all risk-set members through the softmax.

## Breslow Baseline Recovery

After fitting \(\widehat{\beta}\) or \(\widehat{\theta}\), define:

```math
\widehat{\eta}_i=f_{\widehat{\theta}}(z_i).
```

For tied event set \(D_t\), Breslow estimates:

```math
\widehat{\Delta\Lambda}_0(t)
=
\frac{d_t}
{\sum_{j\in R_t}\exp(\widehat{\eta}_j)},
\qquad
d_t=|D_t|.
```

The cumulative baseline hazard is:

```math
\widehat{\Lambda}_0(t)
=
\sum_{s\le t}
\widehat{\Delta\Lambda}_0(s).
```

Then the patient-specific cumulative hazard is:

```math
\widehat{\Lambda}_i(t)
=
\exp(\widehat{\eta}_i)\widehat{\Lambda}_0(t).
```

and survival is:

```math
\widehat{S}_i(t)
=
\exp[-\widehat{\Lambda}_i(t)]
=
\exp[-\exp(\widehat{\eta}_i)\widehat{\Lambda}_0(t)].
```

This is where Cox becomes calibrated, if the baseline estimate and proportional
hazards assumption are adequate.

## Centering The Risk Score

Because:

```math
\eta_i \mapsto \eta_i+c
```

does not change:

```math
\frac{\exp(\eta_i)}
{\sum_{j\in R_i}\exp(\eta_j)},
```

the partial likelihood identifies \(\eta\) only up to an additive constant.

The baseline hazard absorbs the shift:

```math
\lambda_0(t)\exp(\eta_i)
=
[\lambda_0(t)e^c]\exp(\eta_i-c).
```

This matters when comparing risk scores across separately trained models.

## WSI Batch Approximation Problem

The exact risk set is cohort-level:

```math
R_i=\{j:X_j\ge X_i\}.
```

Mini-batch Cox training often replaces \(R_i\) by:

```math
R_i^{(B)}=\{j\in B:X_j\ge X_i\}.
```

This changes the denominator:

```math
\log\sum_{j\in R_i}\exp(\eta_j)
\quad\to\quad
\log\sum_{j\in R_i^{(B)}}\exp(\eta_j).
```

The approximation is biased unless the batch is a representative sample of the
risk set. In WSI, this interacts with:

```text
few events per batch
large GPU memory use per slide
institutional batch effects
class imbalance across cancer subtypes
```

## Dense Summary

```math
\begin{aligned}
\ell(\beta)
&=
\sum_{i:\delta_i=1}
[\beta^\top z_i-\log S^{(0)}(X_i;\beta)],\\
U(\beta)
&=
\sum_{i:\delta_i=1}
[z_i-\bar{z}(X_i;\beta)],\\
\nabla^2\ell(\beta)
&=
-
\sum_{i:\delta_i=1}
V(X_i;\beta),\\
\widehat{\Lambda}_0(t)
&=
\sum_{s\le t}
\frac{d_s}
{\sum_{j\in R_s}\exp(\widehat{\eta}_j)},\\
\widehat{S}_i(t)
&=
\exp[-\exp(\widehat{\eta}_i)\widehat{\Lambda}_0(t)].
\end{aligned}
```
