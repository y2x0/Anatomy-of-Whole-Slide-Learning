# Extended Family Placement

This note places hierarchy, distribution, retrieval-memory, and foundation-latent
representations into the same forward-map language.

The common map is:

```math
S_i
\xrightarrow{\text{tile}}
\{x_{ij},c_{ij}\}_{j=1}^{n_i}
\xrightarrow{E}
H_i
\xrightarrow{\text{Rep}}
\mathcal{X}_i
\xrightarrow{\mathcal{F}}
z_i.
```

The difference is the type of
```math
\mathcal{X}_i
```
.

## Hierarchy

Representation:

```math
\mathcal{X}_i
=
\left(
\{V_i^{(\ell)}\}_{\ell=0}^{L},
\{\pi_i^{(\ell)}\}_{\ell=0}^{L-1},
\{H_i^{(\ell)}\}_{\ell=0}^{L}
\right).
```

Context:

```math
H_i^{(\ell+1)}
=
\mathcal{R}_{\ell}
\left(
\mathcal{C}_{\ell}
\{h_v^{(\ell)}:v\in\mathrm{Ch}(u)\}
\right).
```

Readout:

```math
z_i
=
\psi_\theta
\left(
\mathcal{R}^{(0)}(H_i^{(0)}),
\ldots,
\mathcal{R}^{(L)}(H_i^{(L)})
\right).
```

Survives:

```text
scale-composed tissue summaries
```

Failure mode:

```text
premature compression or wrong parent-child boundaries
```

## Distribution

Representation:

```math
\mathcal{X}_i=\mu_i,
\qquad
\mu_i
=
\frac{1}{n_i}\sum_j\delta_{h_{ij}}.
```

Context:

```math
q_m(h)
=
\mathrm{Assign}(h,c_m)
```

or:

```math
\phi_\theta(h)
=
\text{learned morphology statistic}.
```

Readout:

```math
z_i
=
\left(
\int q_1(h)\,d\mu_i(h),
\ldots,
\int q_M(h)\,d\mu_i(h)
\right).
```

Survives:

```text
morphology prevalence or distribution shape
```

Failure mode:

```text
geometry and rare events can disappear under finite statistics
```

## Retrieval Memory

Representation:

```math
\mathcal{X}_i
=
\left(
q_i,
\mathcal{M}(q_i)
\right),
```

where:

```math
\mathcal{M}
=
\{(k_r,v_r)\}_{r=1}^{N}.
```

Context:

```math
\mathcal{M}(q_i)
=
\{(k_r,v_r):r\in\mathcal{N}_K(i)\}.
```

Readout:

```math
\widehat y_i
=
\mathcal{H}_\theta
\left(
q_i,
\sum_{r\in\mathcal{N}_K(i)}w_{ir}v_r
\right).
```

Survives:

```text
query embedding plus memory neighborhood
```

Failure mode:

```text
retrieved neighbors may encode leakage or nuisance similarity
```

## Foundation Latent

Representation:

```math
\mathcal{X}_i
=
F_{\text{FM}}(S_i)
```

or:

```math
\mathcal{X}_i
=
\{E_{\text{FM}}(x_{ij})\}_{j=1}^{n_i}.
```

Context:

```math
d_{\text{FM}}(a,b)
=
\|F_{\text{FM}}(a)-F_{\text{FM}}(b)\|.
```

Readout:

```math
\widehat y_i
=
\mathcal{H}_\theta(F_{\text{FM}}(S_i)).
```

or, for prompt-based prediction:

```math
p(y=c\mid S_i)
=
\frac{\exp(z_i^\top t_c/\tau)}
{\sum_r\exp(z_i^\top t_r/\tau)}.
```

Survives:

```text
pretraining-shaped latent coordinates
```

Failure mode:

```text
pretraining geometry may not align with the downstream phenotype
```

## Combined Table

| Family | Slide Object | Context Source | Readout | Hidden Assumption |
|---|---|---|---|---|
| Hierarchy | nested tissue units | parent maps and scale | top or multiscale pool | biology composes across chosen scales |
| Distribution | empirical measure | statistic or prototype map | T(\mu) | prevalence captures the signal |
| Retrieval memory | query plus neighbors | external archive | neighbor-weighted context | nearest cases are useful precedent |
| Foundation latent | pretrained embedding | pretraining objective | probe, prompt, or adapter | latent geometry transfers |

The design lesson:

```text
representation is not only what is computed from the slide;
it is also what geometry, memory, or scale structure is allowed to define similarity.
```
