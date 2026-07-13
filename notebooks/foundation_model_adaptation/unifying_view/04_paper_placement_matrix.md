# Paper Placement Matrix

This matrix places pathology foundation models by adaptation mode rather than
by pretraining prestige.

Anchor references: [UNI](https://arxiv.org/abs/2308.15474),
[Virchow](https://www.nature.com/articles/s41591-024-03141-0),
[GigaPath](https://www.nature.com/articles/s41586-024-07441-w),
[CONCH](https://arxiv.org/abs/2307.12914),
[PLIP](https://www.nature.com/articles/s41591-023-02504-3),
[PRISM](https://arxiv.org/abs/2405.10254),
[TITAN](https://arxiv.org/abs/2411.19666).

| Paper/model family | Native object | Common adaptation | Task information enters | Main caveat |
|---|---|---|---|---|
| UNI | patch embedding | MIL head, linear probe, PEFT | readout/head or small updates | patch FM is not a slide model by itself |
| Virchow | patch embedding | MIL head, probe, PEFT | readout/head or small updates | inherited patch geometry may not match target |
| GigaPath | tile encoder plus slide encoder | probe, fine-tune head, slide adaptation | slide head or selected layers | slide statistic is pretrained, not task-specific by default |
| CONCH | image-text embedding | prompting, retrieval, probing | prompt text, memory, head | prompt semantics define the classifier |
| PLIP | image-text embedding | prompting or probing | text prompts and head | web/text supervision can mismatch local pathology |
| PRISM | multimodal slide representation | probing, retrieval, generation-conditioned tasks | report/text-aligned latent space | report semantics may dominate visual truth |
| TITAN | multimodal WSI foundation representation | probe, retrieval, multimodal head | slide/text alignment and head | task may need adaptation beyond alignment |
| Generic LoRA/adapter FM | pretrained encoder | PEFT | low-rank or residual modules | rank and layer placement bottleneck |

## UNI And Virchow

Patch-level foundation encoders produce:

```math
h_{ij}
=
f_{\phi_0}(x_{ij}).
```

A downstream WSI task still needs:

```math
z_i
=
\mathcal{R}_\psi(\{h_{ij}\}_{j=1}^{n_i}).
```

Thus adaptation is often not in the encoder but in the slide aggregator:

```math
S
\to
\psi,\omega.
```

## GigaPath-Style Slide Encoders

A whole-slide foundation model can produce:

```math
z_i^{(0)}
=
F_{\phi_0}(\{h_{ij},c_{ij}\}_{j=1}^{n_i}).
```

Probing asks whether:

```math
P(Y\mid X)
=
P(Y\mid z_i^{(0)}).
```

If not, adaptation must enter the slide encoder, not only the head.

## CONCH And PLIP

Vision-language adaptation often uses:

```math
s_c(X)
=
f_I(X)^\top f_T(p_c).
```

The prompt
```math
p_c
```
is part of the classifier. Prompt engineering is therefore a
supervision design choice, not a cosmetic detail.

## PRISM And TITAN

Multimodal slide foundation models align slide embeddings with reports, text,
or generative objectives. A downstream head uses:

```math
z_i
=
F_{\phi_0}^{\mathrm{slide}}(X_i),
\qquad
\widehat y_i
=
\mathcal{H}_\omega(z_i).
```

The caveat is unit mismatch:

```math
\text{report-level semantics}
\ne
\text{patch-local truth}.
```

## Dense Summary

For every FM paper, report:

```text
native object:
    patch, slide, image-text pair, report-aligned slide

adaptation object:
    head, readout, prompt, memory, PEFT module, full encoder

supervision path:
    where downstream labels enter

failure mode:
    what pretrained geometry assumes but the task violates
```
