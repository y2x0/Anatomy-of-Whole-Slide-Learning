# Additive Survival And Burden

Additive pooling is natural when risk accumulates over tissue burden.

Let patch evidence be:

```math
e_{ij}
=
e_\theta(u_{ij}).
```

An additive Cox score is:

```math
\eta_i
=
\sum_j e_{ij}.
```

The Cox partial likelihood then compares sums of patch evidence across patients.

## Mean Versus Sum Risk

Mean risk:

```math
\eta_i^{\mathrm{mean}}
=
\frac{1}{n_i}\sum_j e_{ij}
```

models average risk density.

Sum risk:

```math
\eta_i^{\mathrm{sum}}
=
\sum_j e_{ij}
```

models total burden.

These make different biological assumptions.

## Tissue Area Correction

If each tile corresponds to area
```math
A
```
, then:

```math
\eta_i
=
\sum_j A e_{ij}
```

approximates an integral over tissue:

```math
\eta_i
\approx
\int_{\mathrm{tissue}} e_\theta(x)\,dx.
```

If tiling density changes across slides, the raw sum can be confounded by
sampling. Area weighting or normalization must match the biological question.

## Prototype Additive Risk

If prototype prevalence is
```math
p_{im}
```
 and tissue amount is
```math
n_i
```
, a burden-like
prototype score is:

```math
\eta_i
=
\sum_m \beta_m n_i p_{im}.
```

A prevalence-like score is:

```math
\eta_i
=
\sum_m \beta_m p_{im}.
```

Again, the difference is total burden versus composition.

## Dense Summary

Additive survival readouts are useful when:

```text
risk accumulates over spatial extent or tissue burden
```

They are risky when:

```text
patch count reflects preprocessing rather than biology
```
