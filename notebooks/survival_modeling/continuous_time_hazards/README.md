# Continuous-Time Hazards

Continuous-time models represent risk as a function of time.

The model may output:

```math
\lambda_\theta(t\mid z_i)
```

or equivalently:

```math
S_\theta(t\mid z_i),
\qquad
f_\theta(t\mid z_i),
\qquad
F_\theta(t\mid z_i).
```

The central identity is:

```math
S(t\mid z)
=
\exp\left(
-\int_0^t \lambda(u\mid z)\,du
\right),
\qquad
f(t\mid z)=\lambda(t\mid z)S(t\mid z).
```

## Why This Matters For WSI

Continuous-time models can, in principle, learn smooth risk curves without
arbitrary time bins.

For WSI, the head is:

```math
(t,z_i)\mapsto \lambda_\theta(t,z_i)
```

or:

```math
(t,z_i)\mapsto f_\theta(t,z_i).
```

That lets morphology interact continuously with time.

## Files

- `01_hazard_survival_density.md`: identities and censored likelihood.
- `02_neural_hazards_and_mixtures.md`: parameterizations.
- `03_wsi_continuous_risk_and_failure_modes.md`: WSI-specific design and risks.
- `04_cumulative_hazard_parameterizations.md`: monotone cumulative hazards,
  splines, positive hazards, and numerical constraints.
- `05_mixture_survival_algebra.md`: mixture likelihoods, posterior components,
  and pathology subtype interpretation.
- `06_time_conditioned_wsi_readouts.md`: time-conditioned attention, gradients,
  and what morphology survives as a function of time.
