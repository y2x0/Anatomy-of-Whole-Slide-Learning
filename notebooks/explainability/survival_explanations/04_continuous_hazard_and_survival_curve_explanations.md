# Continuous Hazard and Survival-Curve Explanations

## 1. Hazard Functional

For continuous hazard `h_i(t)`,

```math
H_i(t)=\int_0^t h_i(u)\,du,
\qquad
S_i(t)=\exp[-H_i(t)].
```

An explanation at time `t` can target instantaneous hazard or accumulated
hazard.

## 2. Functional Derivatives

For a patch-dependent hazard perturbation,

```math
\frac{\delta S_i(t)}{\delta h_i(u)}
=-\mathbf 1\{u\le t\}S_i(t).
```

An early-time hazard change affects survival at all later horizons, while a
change after `t` cannot affect `S_i(t)` under this construction.

## 3. Time-Conditioned Readout

If attention depends on time,

```math
z_i(t)=\sum_j a_{ij}(t)r_{ij},
\qquad
h_i(t)=\mathcal H_t(z_i(t)).
```

The explanation is a function of both patch and time. Reusing one attention map
for the entire curve imposes a strong shared-morphology assumption.

## 4. Curve-Level Distance

For two model outputs, a curve intervention can be summarized by

```math
D_i^{S}
=\int_0^{\tau}
|S_i(t)-S_i^{(-j)}(t)|\,d\mu(t).
```

The measure `mu` determines whether early, late, or clinically selected times
dominate the explanation.

## 5. Numerical Caveat

When the hazard is represented on a grid, the displayed integrals are evaluated
approximately. Time resolution, interpolation, and tail treatment are part of
the explanation definition.

