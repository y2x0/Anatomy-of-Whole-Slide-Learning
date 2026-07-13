# Deep Cox And WSI Readouts

Deep Cox models replace the linear score:

```math
\eta_i = \beta^\top z_i
```

with a neural score:

```math
\eta_i = f_\theta(z_i).
```

The loss is still the Cox partial likelihood:

```math
\mathcal{L}_{\mathrm{Cox}}(\theta)
=
-
\sum_{i:\delta_i=1}
\left[
f_\theta(z_i)
-
\log\sum_{j\in R_i}\exp(f_\theta(z_j))
\right].
```

DeepSurv and Cox-nnet are examples of this move: the survival output remains a
scalar risk score, but the map from covariates to risk becomes nonlinear.

## WSI Pipeline

For a whole slide:

```math
H_i = \{h_{ij}\}_{j=1}^{n_i}.
```

A WSI survival model builds:

```math
z_i = \mathcal{R}(\mathcal{C}(H_i;G_i)).
```

Then:

```math
\eta_i = f_\theta(z_i).
```

The survival loss does not directly see patches. It sees only
```math
z_i
```
.

## Common WSI Readouts

### Mean MIL

```math
z_i = \frac{1}{n_i}\sum_{j=1}^{n_i}h_{ij}.
```

Surviving statistic:

```text
first moment of patch features
```

Risk is assumed to be expressible from average morphology.

### Attention MIL

```math
a_{ij}
=
\frac{\exp(u^\top \tanh(Vh_{ij}))}
{\sum_k \exp(u^\top \tanh(Vh_{ik}))},
\qquad
z_i=\sum_j a_{ij}h_{ij}.
```

Surviving statistic:

```text
weighted first moment
```

This imposes a witness-style inductive-bias assumption: prognostic signal is
concentrated in patches that the attention/readout mechanism can identify.

### Graph Survival

Let patches or regions be graph nodes:

```math
G_i=(V_i,E_i),
\qquad
h_{ij}^{(\ell+1)}
=
\phi\left(
h_{ij}^{(\ell)},
\mathrm{AGG}_{k \in \mathcal{N}(j)}
\psi(h_{ij}^{(\ell)},h_{ik}^{(\ell)})
\right).
```

Then:

```math
z_i = \mathrm{READOUT}(\{h_{ij}^{(L)}\}_{j\in V_i}).
```

Surviving statistic:

```text
contextualized patch or region features after spatial message passing
```

Patch-GCN style models fit here.

### Multimodal Cox Readout

If pathology and genomics are available:

```math
z_i^{\mathrm{path}}=\mathcal{R}_{\mathrm{path}}(H_i),
\qquad
z_i^{\mathrm{omic}}=\mathcal{R}_{\mathrm{omic}}(g_i).
```

A fusion model gives:

```math
z_i = \Phi(z_i^{\mathrm{path}},z_i^{\mathrm{omic}}),
\qquad
\eta_i=f_\theta(z_i).
```

Pathomic Fusion, PORPOISE, MCAT, and SurvPath-style models differ mostly in the
fusion operator
```math
\Phi
```
and the biological structure imposed on omics features.

## What Changes And What Does Not

Changing the WSI encoder changes:

```text
which morphology becomes visible to the survival head
which spatial relationships can affect risk
which patch statistics survive aggregation
```

But Cox training still asks for:

```math
z_i \mapsto \eta_i.
```

The model can use complex slide representations, but the risk object remains
one scalar.

## Key Takeaway

Deep Cox WSI models are best understood as:

```text
complex slide representation learning + scalar risk ranking.
```

The mathematical bottleneck is not only the MIL aggregator. It is also the final
choice to represent risk by a single time-invariant score.
