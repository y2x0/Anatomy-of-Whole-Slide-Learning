# Cause-Specific Hazards

The cause-specific hazard models the instantaneous rate of a particular event
among subjects who have not experienced any event.

For cause
```math
c
```
:

```math
\lambda_c(t\mid z)
=
\lim_{\Delta t\to0}
\frac{
\Pr(t\le T<t+\Delta t,J=c\mid T\ge t,z)
}{\Delta t}.
```

The total hazard is:

```math
\lambda_{\bullet}(t\mid z)=\sum_c\lambda_c(t\mid z).
```

## Cause-Specific Cox Model

A standard cause-specific Cox model is:

```math
\lambda_c(t\mid z)
=
\lambda_{0c}(t)\exp(\eta_c(z)).
```

Each cause has its own baseline hazard and score.

For WSI:

```math
z_i=\mathcal{R}(\mathcal{C}(H_i;G_i)),
\qquad
\eta_{ic}=f_c(z_i).
```

The slide is mapped to a vector of cause-specific risk scores:

```math
\eta_i=(\eta_{i1},\ldots,\eta_{iC}).
```

## Partial Likelihood By Cause

For event type
```math
c
```
, define:

```math
D_c=\{i:\Delta_i=c\}.
```

The risk set at event time
```math
X_i
```
remains:

```math
R_i=\{j:X_j\ge X_i\}.
```

Subjects who have a competing event before
```math
X_i
```
are not in the risk set
because they are no longer event-free.

The partial likelihood for cause
```math
c
```
is:

```math
L_c
=
\prod_{i:\Delta_i=c}
\frac{\exp(\eta_{ic})}
{\sum_{j\in R_i}\exp(\eta_{jc})}.
```

The negative log loss is:

```math
\mathcal{L}_c
=
-
\sum_{i:\Delta_i=c}
\left[
\eta_{ic}
-
\log\sum_{j\in R_i}\exp(\eta_{jc})
\right].
```

The total cause-specific Cox loss is:

```math
\mathcal{L}
=
\sum_{c=1}^{C}\mathcal{L}_c.
```

## What Happens To Competing Events

When fitting cause
```math
c
```
, a subject with event
```math
c'\ne c
```
is treated as event
free until their competing event time and then removed from the risk set.

This is not the same as pretending the competing event never happened. It says:

```text
before the competing event, the subject was at risk for cause c;
after the competing event, the subject cannot experience cause c as first event.
```

## CIF From Cause-Specific Hazards

Cause-specific models estimate
```math
\lambda_c
```
. To get cumulative incidence:

```math
\widehat{F}_c(t\mid z)
=
\int_0^t
\widehat{S}(u\mid z)\widehat{\lambda}_c(u\mid z)\,du.
```

where:

```math
\widehat{S}(u\mid z)
=
\exp\left[
-\sum_{r=1}^{C}
\widehat{\Lambda}_r(u\mid z)
\right].
```

For cause-specific Cox:

```math
\widehat{\Lambda}_r(u\mid z)
=
\widehat{\Lambda}_{0r}(u)\exp(\eta_r(z)).
```

Therefore
```math
F_c
```
depends on all cause-specific models.

## WSI Meaning

If:

```math
\eta_{ic}=w_c^\top z_i,
```

then:

```math
\eta_{ic}
=
w_c^\top\mathcal{R}(\mathcal{C}(H_i;G_i)).
```

The same slide representation may be probed by different cause vectors
```math
w_c
```
. This allows:

```text
morphology predictive of cancer death
morphology predictive of recurrence
morphology unrelated to non-cancer death
```

But if all heads share the same collapsed
```math
z_i
```
, cause-specific morphology
discarded by
```math
\mathcal{R}
```
cannot be recovered.

## Dense Summary

```math
\begin{aligned}
\lambda_c(t\mid z)
&=
\lambda_{0c}(t)\exp(\eta_c(z)),\\
\mathcal{L}_c
&=
-
\sum_{i:\Delta_i=c}
\left[
\eta_{ic}
-
\log\sum_{j\in R_i}\exp(\eta_{jc})
\right],\\
F_c(t\mid z)
&=
\int_0^tS(u\mid z)\lambda_c(u\mid z)\,du.
\end{aligned}
```

Cause-specific hazards are natural for etiologic instantaneous risk. Cumulative
incidence requires recombining all causes.
