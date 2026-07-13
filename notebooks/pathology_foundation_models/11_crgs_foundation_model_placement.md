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

## Pretraining And Downstream Maps

Write the pretraining objective and downstream objective separately:

```math
\phi^{\star}
=
\underset{\phi}{\arg\min}
\mathbb E\left[
\mathcal L_{\mathrm{pre}}\left(
\Phi_{\phi}(X;G,S)
\right)
\right],
```

```math
\psi^{\star},\omega^{\star}
=
\underset{\psi,\omega}{\arg\min}
\mathbb E\left[
\mathcal L_{\mathrm{task}}
\left(
\mathcal H_{\psi}
\left(
\mathcal R_{\omega}
\left(
\mathcal C_{\omega}(\Phi_{\phi^{\star}}(X);G)
\right)
\right),Y
\right].
```

Frozen transfer holds phi fixed. Fine-tuning changes the representation itself.
The distinction matters because a downstream readout can only separate inputs
that remain distinct in the frozen representation:

```math
\Phi_{\phi^{\star}}(X)
=
\Phi_{\phi^{\star}}(X')
\quad\Longrightarrow\quad
\widehat Y(X)=\widehat Y(X')
```

for every deterministic downstream head. This is the formal boundary between a
foundation-model geometry claim and an adaptation claim.

## Contract-Matched Audit

For each family in the table, construct a transformation T that the pretraining
objective declares equivalent and a task-relevant transformation T_star that it
should distinguish. Then measure

```math
\Delta_{\mathrm{inv}}
=
d\left(\Phi(X),\Phi(TX)\right),
\qquad
\Delta_{\mathrm{task}}
=
d\left(\Phi(X),\Phi(T_{\star}X)\right).
```

The desired relationship is contract-dependent, but it should be stated rather
than inferred from a downstream score alone.
