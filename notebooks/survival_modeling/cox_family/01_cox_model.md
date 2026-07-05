# Cox Model

The Cox model is built around one factorization:

```math
\lambda(t\mid z_i)
=
\lambda_0(t)\exp(\eta_i),
\qquad
\eta_i = \beta^\top z_i.
```

Here:

```text
lambda_0(t) = baseline hazard shared across individuals
eta_i       = log-risk score
exp(eta_i) = relative risk
```

The model does not specify \(\lambda_0(t)\) during training. It only learns
which patients should have larger or smaller hazard at every time.

## Relative Risk

For two individuals \(i\) and \(j\):

```math
\frac{\lambda(t\mid z_i)}{\lambda(t\mid z_j)}
=
\frac{\lambda_0(t)\exp(\eta_i)}
{\lambda_0(t)\exp(\eta_j)}
=
\exp(\eta_i-\eta_j).
```

The ratio does not depend on \(t\). This is the proportional hazards assumption.

If patient \(i\) is twice as risky as patient \(j\) at one time, the model says
that same multiplicative relationship holds at all times.

## Survival Curve Under Cox

The cumulative baseline hazard is:

```math
\Lambda_0(t)
=
\int_0^t \lambda_0(u)\,du.
```

For patient \(i\):

```math
\Lambda(t\mid z_i)
=
\int_0^t \lambda_0(u)\exp(\eta_i)\,du
=
\exp(\eta_i)\Lambda_0(t).
```

So:

```math
S(t\mid z_i)
=
\exp[-\Lambda(t\mid z_i)]
=
\exp[-\exp(\eta_i)\Lambda_0(t)].
```

Training can estimate \(\eta_i\) without estimating \(\Lambda_0(t)\), but
calibrated survival curves require a baseline hazard estimate afterward.

## What The Model Learns

The Cox model learns:

```math
z_i \mapsto \eta_i.
```

It does not directly learn:

```text
time-varying effects
absolute event probabilities
separate early-risk and late-risk mechanisms
non-proportional hazards
```

That makes the Cox family a clean readout for WSI embeddings but a narrow risk
representation.

## Whole-Slide Interpretation

Suppose a slide encoder returns:

```math
z_i
=
\sum_{j=1}^{n_i} a_{ij}h_{ij}.
```

A linear Cox head gives:

```math
\eta_i
=
w^\top z_i
=
\sum_{j=1}^{n_i} a_{ij}w^\top h_{ij}.
```

For attention MIL, this exposes the exact survival statistic:

```math
\text{risk score}
=
\text{attention-weighted sum of patch-level risk evidence}.
```

For transformer, graph, or state-space encoders, the same Cox head still
collapses the final slide embedding to one scalar. The context operator changes
the features; the survival head still asks only for an ordering.

## Key Takeaway

The Cox family answers:

```text
Who is riskier than whom?
```

It does not fully answer:

```text
What is the patient's event-time distribution?
```
