# C/R/G/S Foundation Model Placement

## 1. Pretraining Decomposition

```math
H=\Phi_{\phi}(X;G,S),
\qquad
\widetilde H=\mathcal C(H;G,S),
\qquad
z=\mathcal R(\widetilde H;G),
\qquad
\widehat Y=\mathcal H(z).
```

For a foundation model, the pretraining signal shapes `Phi` and sometimes the
slide-level context operator.

| Family | Context | Readout | Surviving statistic | Main bias |
|---|---|---|---|---|
| contrastive | view relation | embedding geometry | pairwise similarity | declared invariances |
| distillation | teacher targets | student representation | teacher-matched function or relation | teacher blind spots |
| masked modeling | visible context | reconstruction head | predictable content | mask and target design |
| vision-language | cross-modal matching | shared embedding | semantic similarity | caption and prompt distribution |
| retrieval | memory metric | nearest-neighbor rule | local cohort evidence | memory composition |
| slide hierarchy | spatial/sequence context | slide encoder | long-context representation | hierarchy bottleneck |

## 2. Adaptation Boundary

The existing adaptation family begins after `Phi` has been learned. A probe,
LoRA module, prompt, or memory bank changes the downstream path without changing
the original pretraining contract unless the encoder is fine-tuned.

## 3. Design Question

Before choosing a foundation model, ask which statistic the downstream task
needs:

```text
rare patch presence, spatial arrangement, tissue distribution, cross-modal
alignment, long-range context, or calibrated patient risk.
```

The largest model is not necessarily the one whose surviving statistic matches
that requirement.

