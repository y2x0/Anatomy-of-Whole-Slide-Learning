# Foundation Model Adaptation

This notebook family asks:

```text
How is task information injected into a pretrained pathology model?
```

A foundation model supplies a pretrained map:

```math
f_{\phi_0}:x\mapsto h,
```

or a slide-level map:

```math
F_{\phi_0}:\{x_j,c_j\}_{j=1}^{n}\mapsto z.
```

Adaptation chooses which variables are allowed to move:

```math
\phi_0
\to
\phi_0+\Delta_\eta,
```

and which variables remain frozen. The central question is not merely "which
method performs best?" It is:

```text
where do downstream labels change the computation?
```

## Folder Map

```text
00_problem_setup.md
    pretrained maps, task losses, parameter partitions, and adaptation axes

frozen_feature_pipelines/
    patch and slide encoders used as fixed maps
    task information enters the readout or head

linear_probing/
    linear/logistic/Cox probes on frozen representations
    separability, calibration, and domain shift

parameter_efficient_tuning/
    LoRA, adapters, bias/norm tuning, and rank-limited updates
    task information enters a restricted parameter subspace

prompt_tuning/
    text prompts, soft prompts, visual prompts, and prompt ensembles
    task information enters tokens rather than most weights

full_finetuning/
    end-to-end gradient flow into the encoder
    drift, forgetting, and small-cohort overfitting

retrieval_memory_adaptation/
    kNN, prototypes, memory banks, and retrieval-augmented heads
    task information enters stored examples or external memories

unifying_view/
    task-information injection matrix
    parameter partition and gradient-flow view
    paper placement and decision checklist
```

## C/R/G/S Placement

Foundation adaptation modifies the repository-wide decomposition:

```math
\widetilde H_i
=
\mathcal{C}_{\eta}(H_i;G_i,S_i;\phi_0),
\qquad
z_i
=
\mathcal{R}_{\eta}(\widetilde H_i;G_i,S_i;\phi_0),
\qquad
\widehat y_i
=
\mathcal{H}_{\eta}(z_i).
```

The pretrained parameters $\phi_0$ may be fixed or adapted. The downstream
parameters $\eta$ may live in the head, readout, prompt, low-rank update,
adapter, memory bank, or the full encoder.

## Anchor Methods And Papers

- UNI and Virchow: patch-level pathology foundation encoders used with probes,
  MIL heads, or task-specific adaptation.
- GigaPath: tile encoder plus slide encoder for whole-slide foundation
  representations.
- CONCH and PLIP: vision-language pathology models adapted through text prompts,
  retrieval, or downstream heads.
- PRISM and TITAN: multimodal slide-level foundation models with report/text
  alignment and downstream probing or adaptation.
- LoRA, adapters, prompt tuning, and linear probing as general adaptation
  machinery.

## Dense Summary

Foundation adaptation is a constrained optimization problem:

```math
\min_{\eta}
\mathbb{E}
\left[
\ell
\left(
T_{\eta,\phi_0}(X),
Y
\right)
\right]
\quad
\text{subject to }
\Delta_\eta\in\mathcal{A}.
```

The adaptation family $\mathcal{A}$ defines the inductive bias:

```text
linear probe:
    only the decision boundary moves

prompt tuning:
    the query into the pretrained model moves

LoRA/adapters:
    a low-dimensional parameter subspace moves

full fine-tuning:
    the representation itself moves

retrieval:
    the memory and metric define the task adaptation
```

The failure mode is always the same in different clothes: the task-relevant
pathology direction may not lie in the part of the model you allowed to move.
