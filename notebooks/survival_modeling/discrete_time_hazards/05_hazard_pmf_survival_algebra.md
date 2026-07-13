# Hazard, PMF, And Survival Algebra

Discrete-time survival models have several equivalent output parameterizations.
The algebra matters because each parameterization induces different constraints.

Let intervals be:

```math
I_k=(\tau_{k-1},\tau_k],
\qquad k=1,\ldots,K.
```

Define:

```math
h_k
=
\Pr(T\in I_k\mid T>\tau_{k-1},z),
```

```math
p_k
=
\Pr(T\in I_k\mid z),
```

```math
S_k
=
\Pr(T>\tau_k\mid z).
```

Suppress
```math
i
```
for readability.

## Hazard To Survival

Survival through interval
```math
k
```
:

```math
S_k
=
\prod_{\ell=1}^k(1-h_\ell).
```

Set:

```math
S_0=1.
```

Then:

```math
S_k=S_{k-1}(1-h_k).
```

## Hazard To PMF

The event probability in interval
```math
k
```
is:

```math
p_k
=
\Pr(T>\tau_{k-1}\mid z)
\Pr(T\in I_k\mid T>\tau_{k-1},z)
=
S_{k-1}h_k.
```

Therefore:

```math
p_k
=
h_k\prod_{\ell=1}^{k-1}(1-h_\ell).
```

The remaining tail mass after
```math
K
```
intervals is:

```math
p_{>K}=S_K=\prod_{\ell=1}^K(1-h_\ell).
```

So:

```math
\sum_{k=1}^Kp_k+p_{>K}=1.
```

## PMF To Survival

If a model predicts
```math
p_1,\ldots,p_K,p_{>K}
```
, then:

```math
S_k
=
\Pr(T>\tau_k\mid z)
=
\sum_{\ell=k+1}^{K}p_\ell+p_{>K}.
```

Equivalently:

```math
S_k=1-\sum_{\ell=1}^{k}p_\ell.
```

## PMF To Hazard

Since:

```math
p_k=S_{k-1}h_k,
```

we get:

```math
h_k
=
\frac{p_k}{S_{k-1}}
=
\frac{p_k}{1-\sum_{\ell=1}^{k-1}p_\ell}.
```

This is valid only when
```math
S_{k-1}>0
```
.

## Cumulative Incidence

The cumulative event probability through time
```math
\tau_k
```
is:

```math
F_k
=
\Pr(T\le\tau_k\mid z)
=
1-S_k
=
\sum_{\ell=1}^{k}p_\ell.
```

In hazard form:

```math
F_k
=
1-\prod_{\ell=1}^{k}(1-h_\ell).
```

## Logit Hazard Parameterization

Let:

```math
g_k=w_k^\top z+b_k,
\qquad
h_k=\sigma(g_k).
```

Then:

```math
\log(1-h_k)
=
\log\sigma(-g_k).
```

For an event in interval
```math
r
```
, the log likelihood is:

```math
\log p_r
=
\sum_{k<r}\log\sigma(-g_k)
+
\log\sigma(g_r).
```

For censoring after interval
```math
r
```
:

```math
\log S_r
=
\sum_{k\le r}\log\sigma(-g_k).
```

## Softmax PMF Parameterization

Alternatively:

```math
(p_1,\ldots,p_K,p_{>K})
=
\mathrm{softmax}(u_1,\ldots,u_K,u_{>K}).
```

This automatically normalizes the event-time distribution.

But the hazard implied by the PMF is:

```math
h_k
=
\frac{\exp(u_k)}
{\sum_{\ell=k}^{K}\exp(u_\ell)+\exp(u_{>K})}.
```

Thus a softmax PMF parameterization creates hazards through suffix
normalization.

## Monotonicity Constraint

Survival must be nonincreasing:

```math
S_0\ge S_1\ge\cdots\ge S_K\ge0.
```

Hazard parameterization guarantees this if:

```math
0\le h_k\le1.
```

PMF parameterization guarantees this if:

```math
p_k\ge0,
\qquad
\sum_kp_k+p_{>K}=1.
```

A direct neural output for
```math
S_k
```
must enforce monotonicity separately.

## WSI Consequence

If:

```math
h_k=\sigma(w_k^\top z+b_k),
```

then all survival quantities are nonlinear functions of the same time-indexed
linear probes:

```math
S_k
=
\prod_{\ell=1}^{k}
\sigma[-(w_\ell^\top z+b_\ell)].
```

The cumulative incidence is:

```math
F_k
=
1-
\prod_{\ell=1}^{k}
\sigma[-(w_\ell^\top z+b_\ell)].
```

Thus a WSI discrete hazard head is not just
```math
K
```
independent classifiers. The
survival curve couples all earlier hazards multiplicatively.

## Dense Summary

```math
\begin{aligned}
S_0&=1,\\
S_k&=\prod_{\ell=1}^{k}(1-h_\ell),\\
p_k&=S_{k-1}h_k,\\
F_k&=1-S_k=\sum_{\ell=1}^{k}p_\ell,\\
h_k&=\frac{p_k}{1-\sum_{\ell<k}p_\ell}.
\end{aligned}
```

The choice of output parameterization determines which constraints are automatic
and which must be enforced.
