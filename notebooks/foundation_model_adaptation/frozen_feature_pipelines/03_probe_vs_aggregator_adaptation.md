# Probe Versus Aggregator Adaptation

There are two different ways to use frozen foundation features.

Head-only probing:

```math
h_{ij}
=
f_{\phi_0}(x_{ij}),
\qquad
z_i
=
z_i^{(0)},
\qquad
\widehat y_i
=
\mathcal{H}_\omega(z_i).
```

Aggregator adaptation:

```math
h_{ij}
=
f_{\phi_0}(x_{ij}),
\qquad
z_i
=
\mathcal{R}_\psi(\{h_{ij}\}_{j=1}^{n_i}),
\qquad
\widehat y_i
=
\mathcal{H}_\omega(z_i).
```

The first assumes the slide statistic already exists. The second assumes the
patch geometry exists, but the slide statistic must be learned.

## Function Classes

Head-only probing learns:

```math
\mathcal{F}_{\mathrm{probe}}
=
\{x\mapsto g_\omega(F_{\phi_0}(x))\}.
```

Aggregator adaptation learns:

```math
\mathcal{F}_{\mathrm{agg}}
=
\{X\mapsto g_\omega(\mathcal{R}_\psi(\{f_{\phi_0}(x_j)\}_j))\}.
```

Usually:

```math
\mathcal{F}_{\mathrm{probe}}
\subset
\mathcal{F}_{\mathrm{agg}},
```

because the readout can be task-specific.

## When They Differ

Two slides can have the same frozen mean but different upper-tail evidence:

```math
\frac{1}{n}\sum_j h_j
=
\frac{1}{m}\sum_k h'_k,
```

while:

```math
\max_j w^\top h_j
\ne
\max_k w^\top h'_k.
```

A head on the mean cannot separate them. An adapted readout can.

## Dense Summary

The phrase "using a foundation model" is underspecified. It may mean:

```text
probe:
    the slide representation is fixed

aggregator adaptation:
    patch embeddings are fixed but the slide statistic is learned
```

These are different mathematical hypotheses about what pretraining already
solved.
