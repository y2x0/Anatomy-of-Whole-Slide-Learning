# MIL Survival

Multiple-instance learning treats the slide as a bag:

```math
H_i=\{h_{ij}\}_{j=1}^{n_i}.
```

Survival MIL differs from classification MIL because the bag label is censored
time-to-event data.

## Mean MIL Survival

Readout:

```math
z_i=\frac{1}{n_i}\sum_{j=1}^{n_i}h_{ij}.
```

Cox head:

```math
\eta_i=w^\top z_i.
```

Then:

```math
\eta_i
=
\frac{1}{n_i}\sum_jw^\top h_{ij}.
```

Surviving statistic:

```text
average patch risk evidence
```

Failure:

```text
rare prognostic morphology is diluted
```

## Attention MIL Survival

Attention:

```math
a_{ij}
=
\frac{\exp(u^\top\tanh(Vh_{ij}))}
{\sum_{\ell}\exp(u^\top\tanh(Vh_{i\ell}))}.
```

Readout:

```math
z_i=\sum_ja_{ij}h_{ij}.
```

Cox risk:

```math
\eta_i=w^\top z_i
=
\sum_ja_{ij}w^\top h_{ij}.
```

Surviving statistic:

```text
attention-weighted first moment of patch risk evidence
```

If the output is discrete hazards:

```math
h_{ik}=\sigma(w_k^\top z_i+b_k),
```

then the same attention summary feeds every time bin unless attention depends on
time.

## Additive Survival MIL

An additive risk model writes:

```math
\eta_i=\sum_jg_\theta(h_{ij}).
```

or normalized:

```math
\eta_i=\frac{1}{n_i}\sum_jg_\theta(h_{ij}).
```

This gives direct patch-level contributions:

```math
e_{ij}=g_\theta(h_{ij}).
```

It is interpretable but assumes the slide risk decomposes additively across
patches.

## Prototype Survival

Let:

```math
p_{im}
=
\frac{1}{n_i}\sum_jq_{ijm}
```

be the prevalence of prototype
```math
m
```
.

Cox:

```math
\eta_i=\sum_m\beta_mp_{im}.
```

Discrete hazards:

```math
h_{ik}=\sigma\left(b_k+\sum_m\beta_{mk}p_{im}\right).
```

Continuous hazard:

```math
\lambda_i(t)
=
\mathrm{softplus}
\left[
b(t)+\sum_m\beta_m(t)p_{im}
\right].
```

This treats the slide as a morphology distribution.

## MIL With Censored Loss

For any MIL readout
```math
z_i
```
, the loss may be:

```math
\ell_i
=
-\delta_i\log\lambda_i(X_i)
+\int_0^{X_i}\lambda_i(u)\,du
```

or:

```math
\ell_i
=
-
\sum_km_{ik}
[y_{ik}\log h_{ik}+(1-y_{ik})\log(1-h_{ik})]
```

or Cox:

```math
\ell_i
=
-
\delta_i
\left[
\eta_i-\log\sum_{j\in R_i}\exp(\eta_j)
\right].
```

The aggregator determines what morphology the loss can shape.

## Dense Summary

```text
mean MIL:
    first moment

attention MIL:
    weighted first moment

additive MIL:
    sum of patch evidence

prototype MIL:
    morphology distribution
```

The survival head changes the risk object, but the MIL readout controls which
slide statistics survive.
