# End-To-End And Foundation Features

WSI survival models differ in whether the patch encoder is frozen.

## Frozen Feature Pipeline

Most WSI survival models use:

```math
h_{ij}=E_0(x_{ij}),
```

where $E_0$ is fixed. Then train:

```math
z_i=\mathcal{R}_{\theta}(\mathcal{C}_{\theta}(H_i)),
\qquad
\widehat{Y}_i^{\operatorname{surv}}=\mathcal{H}_{\theta}(z_i).
```

Advantages:

```text
fits GPU memory
works with small survival cohorts
separates feature extraction from survival learning
```

Limitation:

```text
the encoder never receives survival gradients
```

## End-To-End Pipeline

End-to-end training makes:

```math
h_{ij}=E_{\theta}(x_{ij}).
```

The loss gradient reaches patch pixels:

```math
\nabla_{\theta_E}\mathcal{L}
=
\sum_{i,j}
\frac{\partial\mathcal{L}}{\partial h_{ij}}
\frac{\partial E_{\theta}(x_{ij})}{\partial\theta_E}.
```

This can adapt features to prognosis but is difficult because:

```text
gigapixel slides
many patches per slide
small event counts
risk-set coupling
heavy censoring
```

Recent end-to-end WSI survival work tries to reduce this computational barrier.

## Foundation Model Features

Pathology foundation models produce:

```math
h_{ij}=E_{\operatorname{FM}}(x_{ij}).
```

Survival learning becomes:

```math
E_{\operatorname{FM}}
\to
\mathcal{C}_{\theta}
\to
\mathcal{R}_{\theta}
\to
\mathcal{H}_{\theta}.
```

The question is whether the foundation embedding contains prognostic axes:

```text
grade
necrosis
invasion
immune morphology
stroma
molecular correlates
```

If not, the survival head cannot recover them.

## Linear Probe Vs Adaptation

Linear Cox probe:

```math
\eta_i=w^\top z_i.
```

Adapter:

```math
\widetilde{h}_{ij}=h_{ij}+A_{\theta}(h_{ij}).
```

LoRA-style update:

```math
W'=W+BA.
```

Prompt/tuning style adaptation:

```math
\widetilde{H}_i=E_{\operatorname{FM}}(x_i;P_{\theta}).
```

Survival data are often too small for full fine-tuning, so parameter-efficient
adaptation is attractive.

## Baseline Hazard Issue

If a foundation model changes the risk-score scale:

```math
\eta_i'=a\eta_i+b,
```

Cox ordering may be similar, but calibrated survival changes:

```math
\widehat{S}_i(t)
=
\exp[-\exp(\eta_i')\widehat{\Lambda}_0(t)].
```

Thus foundation-model comparisons should not rely only on C-index.

## Dense Summary

```text
frozen features:
    stable but no survival adaptation

end-to-end:
    expressive but expensive and event-limited

foundation model:
    strong morphology prior, uncertain prognosis sufficiency

PEFT:
    compromise between adaptation and overfitting
```

The mathematical question is where survival gradients are allowed to flow.
