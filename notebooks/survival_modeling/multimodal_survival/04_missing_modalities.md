# Missing Modalities

Multimodal survival data are often incomplete.

Let:

```math
M_i^m=\mathbf{1}[\text{modality }m\text{ is observed for patient }i].
```

Observed modalities:

```math
\mathcal{O}_i=\{m:M_i^m=1\}.
```

## Masked Fusion

A masked additive fusion is:

```math
z_i
=
\frac{
\sum_mM_i^mW_mz_i^m
}{
\sum_mM_i^m
}.
```

This is simple but assumes missing modalities can be ignored after rescaling.

## Modality Dropout

During training, randomly mask modalities:

```math
\widetilde{M}_i^m
=
M_i^mB_i^m,
\qquad
B_i^m\sim\operatorname{Bernoulli}(q_m).
```

Train:

```math
z_i=\Phi(\{\widetilde{M}_i^mz_i^m\}_m).
```

This reduces dependence on any single modality.

## Product Of Experts With Missingness

If each modality defines cumulative hazard:

```math
\Lambda_i^m(t),
```

combine observed modalities:

```math
\Lambda_i(t)
=
\sum_mM_i^m\alpha_m\Lambda_i^m(t).
```

Then:

```math
S_i(t)=\exp[-\Lambda_i(t)].
```

Missing modalities simply remove experts.

## Generative Imputation

A model may infer missing modality $g$ from pathology:

```math
q_\phi(z_i^g\mid z_i^p).
```

Survival prediction marginalizes:

```math
p(T_i\mid z_i^p)
=
\int
p_\theta(T_i\mid z_i^p,z_i^g)
q_\phi(z_i^g\mid z_i^p)
\,dz_i^g.
```

This is principled but depends on the imputation model.

## Missingness Mechanism

Missingness may be informative:

```math
M_i^m\not\perp T_i\mid z_i.
```

For example, genomic testing may be ordered for advanced or unusual cases. Then
missingness itself can carry risk information.

Include missingness indicators:

```math
z_i=\Phi(\{M_i^mz_i^m\}_m,\{M_i^m\}_m).
```

## Dense Summary

```text
masked fusion:
    use what exists

modality dropout:
    train robustness

product of experts:
    additive cumulative hazards from observed modalities

imputation:
    infer missing modality distribution

missingness indicators:
    model informative missingness
```

Survival missingness is not just an engineering problem; it can change the risk
target.
