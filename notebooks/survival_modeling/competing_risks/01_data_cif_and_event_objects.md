# Data, CIFs, And Event Objects

In ordinary survival analysis, one event time $T$ is modeled. In competing
risks, there are latent event times:

```math
T^{(1)},T^{(2)},\ldots,T^{(C)}.
```

The observed event time is:

```math
T=\min_{c}T^{(c)}.
```

The observed event type is:

```math
J=\arg\min_{c}T^{(c)}.
```

With censoring:

```math
X=\min(T,C),
\qquad
\delta=\mathbf{1}[T\le C],
\qquad
\Delta=\delta J.
```

The value $\Delta=0$ denotes censoring.

## Cause-Specific Probability Mass

The event-time/cause distribution is:

```math
p_c(t\mid z)\,dt
=
\Pr(t\le T<t+dt,J=c\mid z).
```

The cause-specific cumulative incidence function is:

```math
F_c(t\mid z)
=
\Pr(T\le t,J=c\mid z)
=
\int_0^t p_c(u\mid z)\,du.
```

The all-cause survival function is:

```math
S(t\mid z)
=
\Pr(T>t\mid z)
=
1-\sum_{c=1}^{C}F_c(t\mid z).
```

The event-type probabilities plus survival must satisfy:

```math
\sum_{c=1}^{C}F_c(t\mid z)+S(t\mid z)=1.
```

## Cause-Specific Hazard

The cause-specific hazard is:

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
\lambda_{\bullet}(t\mid z)
=
\sum_{c=1}^{C}\lambda_c(t\mid z).
```

Then:

```math
S(t\mid z)
=
\exp\left[
-\int_0^t\lambda_{\bullet}(u\mid z)\,du
\right].
```

The cause-specific event density is:

```math
p_c(t\mid z)
=
S(t\mid z)\lambda_c(t\mid z).
```

Therefore:

```math
F_c(t\mid z)
=
\int_0^t
S(u\mid z)\lambda_c(u\mid z)\,du.
```

This identity is central: the cumulative incidence for cause $c$ depends on
all causes through $S(u\mid z)$, not only on $\lambda_c$.

## Why CIF Is Not One Minus Survival

In a single-risk setting:

```math
F(t\mid z)=1-S(t\mid z).
```

In competing risks:

```math
F_c(t\mid z)\ne 1-S_c(t\mid z)
```

unless $S_c$ is carefully defined in a counterfactual world where competing
events are removed. That counterfactual object is not the observed event process.

The observed all-cause survival is:

```math
S(t\mid z)=1-\sum_cF_c(t\mid z).
```

Each $F_c$ is a part of the event distribution.

## WSI Label Interpretation

For WSI prognosis, competing risks can correspond to:

```text
cancer-specific death
non-cancer death
recurrence
treatment toxicity
progression
second primary cancer
```

A slide may be predictive of one event type but not another. For example:

```text
tumor morphology may predict cancer-specific death
clinical frailty may predict non-cancer death
immune patterns may predict recurrence under treatment
```

Thus the model should specify whether it outputs:

```math
\lambda_c(t\mid z)
```

or:

```math
F_c(t\mid z).
```

## Dense Summary

```math
\begin{aligned}
X&=\min(T,C),\\
\Delta&=\delta J,\qquad \delta=\mathbf{1}[T\le C],\\
\lambda_{\bullet}(t\mid z)&=\sum_{c=1}^{C}\lambda_c(t\mid z),\\
S(t\mid z)&=\exp\left[-\int_0^t\lambda_{\bullet}(u\mid z)\,du\right],\\
F_c(t\mid z)&=\int_0^tS(u\mid z)\lambda_c(u\mid z)\,du.
\end{aligned}
```

Competing risks are about a vector-valued event distribution, not many
independent binary survival problems.
