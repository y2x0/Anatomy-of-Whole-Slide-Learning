# Subdistribution Hazards And Fine-Gray

Fine-Gray regression targets the cumulative incidence function more directly
than cause-specific hazards.

For cause $c$, define:

```math
F_c(t\mid z)=\Pr(T\le t,J=c\mid z).
```

The subdistribution survival for cause $c$ is:

```math
S_c^{\text{sub}}(t\mid z)
=
1-F_c(t\mid z).
```

The subdistribution hazard is:

```math
\alpha_c(t\mid z)
=
-
\frac{d}{dt}
\log[1-F_c(t\mid z)].
```

Equivalently:

```math
\alpha_c(t\mid z)
=
\frac{dF_c(t\mid z)/dt}{1-F_c(t\mid z)}.
```

This is a hazard-like object for the CIF, not the instantaneous biological event
rate among event-free subjects.

## Fine-Gray Proportional Subdistribution Hazards

The Fine-Gray model is:

```math
\alpha_c(t\mid z)
=
\alpha_{0c}(t)\exp(\gamma_c^\top z).
```

Then:

```math
1-F_c(t\mid z)
=
\exp[-A_{0c}(t)\exp(\gamma_c^\top z)],
```

where:

```math
A_{0c}(t)=\int_0^t\alpha_{0c}(u)\,du.
```

Thus:

```math
F_c(t\mid z)
=
1-
\exp[-A_{0c}(t)\exp(\gamma_c^\top z)].
```

The model directly gives a regression structure for the CIF of one cause.

## Modified Risk Set

Fine-Gray keeps subjects who experience competing events in the risk set for the
subdistribution process. Intuitively:

```text
they cannot experience the event of interest anymore, but they remain part of
the denominator defining the probability mass not yet assigned to cause c.
```

This is why the subdistribution hazard is not the same as a cause-specific
hazard.

A schematic weighted risk set is:

```math
R_i^{\text{FG}}
=
\{j:X_j\ge X_i\}
\cup
\{j:X_j<X_i,\Delta_j\ne 0,\Delta_j\ne c\},
```

with censoring weights used to handle censoring. The exact estimating equations
use inverse probability of censoring weights.

## Interpretation

Cause-specific hazard:

```math
\lambda_c(t\mid z)
=
\text{instantaneous rate of cause c among event-free subjects}.
```

Subdistribution hazard:

```math
\alpha_c(t\mid z)
=
\text{instantaneous rate of accumulating CIF mass for cause c}.
```

The Fine-Gray coefficient describes effects on:

```math
F_c(t\mid z),
```

not on the immediate biological rate of cause $c$ among those still
event-free.

## Why This Distinction Matters

A covariate can increase the cause-specific hazard for cause $c$, but also
increase the competing event hazard so strongly that the CIF for cause $c$
decreases.

Mathematically:

```math
F_c(t\mid z)
=
\int_0^t
\exp\left[
-\int_0^u\sum_r\lambda_r(s\mid z)\,ds
\right]
\lambda_c(u\mid z)\,du.
```

Changing any $\lambda_r$ changes $F_c$.

## WSI Interpretation

For WSI survival:

```math
\alpha_c(t\mid S_i)
=
\alpha_{0c}(t)
\exp(f_c(\mathcal{R}(\mathcal{C}(H_i)))).
```

This says the slide representation directly orders cumulative incidence for
event $c$.

That can be the right target for clinical prediction:

```text
probability of recurrence by 3 years before another terminal event
probability of cancer-specific death by 5 years
```

It is less direct for mechanistic interpretation of instantaneous disease
processes.

## Dense Summary

```math
\begin{aligned}
S_c^{\text{sub}}(t\mid z)
&=1-F_c(t\mid z),\\
\alpha_c(t\mid z)
&=
\frac{dF_c(t\mid z)/dt}{1-F_c(t\mid z)},\\
\alpha_c(t\mid z)
&=
\alpha_{0c}(t)\exp(\gamma_c^\top z),\\
F_c(t\mid z)
&=
1-\exp[-A_{0c}(t)\exp(\gamma_c^\top z)].
\end{aligned}
```

Fine-Gray is CIF regression. It should not be read as ordinary biological
hazard regression.
