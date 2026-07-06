# WSI Survival Pipeline

A whole-slide image is too large to enter most survival models directly. The
standard pipeline is:

```math
S_i
\xrightarrow{\text{tile}}
\{x_{ij}\}_{j=1}^{n_i}
\xrightarrow{E}
H_i=\{h_{ij}\}_{j=1}^{n_i}.
```

Here:

```math
h_{ij}=E(x_{ij})\in\mathbb{R}^{d}.
```

The slide representation is:

```math
z_i=\mathcal{R}(\mathcal{C}(H_i;G_i)).
```

The survival head maps:

```math
z_i\mapsto \mathcal{Y}_i^{\text{surv}}.
```

where:

```math
\mathcal{Y}_i^{\text{surv}}
\in
\{\eta_i,h_i,\lambda_i,S_i,p_i,F_i\}.
```

## Four Operators

Context operator:

```math
\widetilde{H}_i=\mathcal{C}(H_i;G_i).
```

Readout operator:

```math
z_i=\mathcal{R}(\widetilde{H}_i).
```

Risk head:

```math
\mathcal{Y}_i^{\text{surv}}
=
\mathcal{H}_{\theta}(z_i).
```

Loss:

```math
\mathcal{L}
=
\sum_i
\ell(\mathcal{Y}_i^{\text{surv}},X_i,\delta_i).
```

## Where Survival Enters

Survival supervision can enter at different points.

Final head only:

```math
H_i\to z_i\to \eta_i.
```

Time-conditioned readout:

```math
H_i\to z_i(t)\to \lambda_i(t).
```

Cause-conditioned readout:

```math
H_i\to z_{ic}\to F_{ic}(t).
```

Instance-level auxiliary supervision:

```math
h_{ij}\to r_{ij},
\qquad
z_i=\mathcal{R}(\{h_{ij},r_{ij}\}).
```

Most WSI survival papers use final-head supervision only.

## Survival Labels

For each patient:

```math
X_i=\min(T_i,C_i),
\qquad
\delta_i=\mathbf{1}[T_i\le C_i].
```

If there are multiple slides per patient, the patient representation may be:

```math
z_i^{\text{patient}}
=
\mathcal{R}_{\text{patient}}
(\{z_{is}\}_{s=1}^{m_i}).
```

The survival label belongs to the patient, not the tile.

## Dense Summary

```math
\begin{aligned}
h_{ij}&=E(x_{ij}),\\
\widetilde{H}_i&=\mathcal{C}(H_i;G_i),\\
z_i&=\mathcal{R}(\widetilde{H}_i),\\
\widehat{Y}_i^{\text{surv}}&=\mathcal{H}_{\theta}(z_i),\\
\mathcal{L}&=\sum_i\ell(\widehat{Y}_i^{\text{surv}},X_i,\delta_i).
\end{aligned}
```

The survival model is only as rich as the slide statistic that survives
$\mathcal{C}$ and $\mathcal{R}$.
