# Neural Competing Risks And DeepHit

Neural competing-risk models often parameterize the full event-time/cause
distribution directly.

Discretize time into
```math
K
```
bins. A model outputs:

```math
p_{ikc}
=
\Pr(T_i\in I_k,J_i=c\mid z_i).
```

The probabilities satisfy:

```math
p_{ikc}\ge0,
\qquad
\sum_{k=1}^{K}\sum_{c=1}^{C}p_{ikc}+p_{i,>K}=1.
```

A softmax over
```math
KC+1
```
cells enforces this:

```math
(p_{i11},\ldots,p_{iKC},p_{i,>K})
=
\mathrm{softmax}(u_i).
```

## CIF And Survival

The cause-specific CIF is:

```math
F_{ic}(\tau_k)
=
\sum_{\ell=1}^{k}p_{i\ell c}.
```

The all-cause survival is:

```math
S_i(\tau_k)
=
1-\sum_{c=1}^{C}F_{ic}(\tau_k)
=
1-\sum_{\ell=1}^{k}\sum_{c=1}^{C}p_{i\ell c}.
```

The model predicts the full discrete competing-risk distribution.

## Likelihood

If subject
```math
i
```
 has event
```math
c
```
 in interval
```math
r
```
:

```math
\mathcal{L}_i^{\text{event}}
=
-\log p_{irc}.
```

If subject
```math
i
```
 is censored in interval
```math
r
```
, the observed event time exceeds
the censoring time:

```math
\mathcal{L}_i^{\text{cens}}
=
-\log
\left[
\sum_{\ell>r}\sum_{c=1}^{C}p_{i\ell c}
+p_{i,>K}
\right].
```

Equivalently:

```math
\mathcal{L}_i^{\text{cens}}
=
-\log S_i(\tau_r).
```

## DeepHit-Style Objective

DeepHit combines a likelihood term with a ranking term.

The likelihood term is:

```math
\mathcal{L}_{1}
=
\sum_i
\left[
\mathbf{1}[\Delta_i\ne0](-\log p_{i\kappa_i\Delta_i})
+
\mathbf{1}[\Delta_i=0](-\log S_i(X_i))
\right].
```

A cause-specific ranking term compares cumulative incidence for comparable
pairs:

```math
\mathcal{L}_{2}
=
\sum_{c=1}^{C}
\sum_{i:\Delta_i=c}
\sum_{j:X_j>X_i}
\omega_{ij}
\exp\left(
-
\frac{
F_{ic}(X_i)-F_{jc}(X_i)
}{\sigma}
\right).
```

The total loss is:

```math
\mathcal{L}
=
\alpha\mathcal{L}_{1}
+(1-\alpha)\mathcal{L}_{2}.
```

Different implementations vary in the exact pair weighting and smoothing
function, but the structure is likelihood plus ranking over CIFs.

## Relation To Hazard Models

A PMF model outputs
```math
p_{kc}
```
. A cause-specific discrete hazard model outputs:

```math
h_{kc}
=
\Pr(T\in I_k,J=c\mid T>\tau_{k-1},z).
```

The total interval hazard is:

```math
h_{k\bullet}=\sum_c h_{kc}.
```

Survival to the start of interval
```math
k
```
is:

```math
S_{k-1}
=
\prod_{\ell<k}(1-h_{\ell\bullet}).
```

Then:

```math
p_{kc}=S_{k-1}h_{kc}.
```

Conversely:

```math
h_{kc}
=
\frac{p_{kc}}
{1-\sum_{\ell<k}\sum_rp_{\ell r}}.
```

PMF and hazard parameterizations are connected, but they induce different
network constraints and gradients.

## WSI Head

For WSI:

```math
z_i=\mathcal{R}(\mathcal{C}(H_i;G_i)).
```

A simple PMF head is:

```math
u_i=Wz_i+b,
\qquad
W\in\mathbb{R}^{(KC+1)\times d}.
```

Then:

```math
p_i=\mathrm{softmax}(u_i).
```

This makes each time-cause cell a probe of the same slide representation.

A richer design uses cause-time-specific readouts:

```math
z_{ikc}
=
\sum_j a_{ijkc}h_{ij},
\qquad
p_{ikc}
=
\mathrm{softmax}_{k,c}(w_{kc}^\top z_{ikc}).
```

Now different event types and horizons can attend to different morphology.

## Dense Summary

```math
\begin{aligned}
p_{ikc}&=\Pr(T_i\in I_k,J_i=c\mid z_i),\\
F_{ic}(\tau_k)&=\sum_{\ell\le k}p_{i\ell c},\\
S_i(\tau_k)&=1-\sum_{\ell\le k}\sum_cp_{i\ell c},\\
\mathcal{L}_i&=
-\mathbf{1}[\Delta_i\ne0]\log p_{i\kappa_i\Delta_i}
-\mathbf{1}[\Delta_i=0]\log S_i(X_i).
\end{aligned}
```

Deep competing-risk networks are clearest when treated as models of the joint
discrete distribution over time and event type.
