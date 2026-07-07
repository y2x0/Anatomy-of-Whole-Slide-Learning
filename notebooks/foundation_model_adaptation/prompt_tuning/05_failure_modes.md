# Failure Modes

## Prompt Mismatch

If a prompt encodes the wrong concept:

```math
v_c
=
f_T(p_c)
```

points toward the wrong text direction. The model can fail even if the visual
embedding contains the correct morphology.

## Local-Global Mismatch

A slide label may be global:

```math
Y_{\mathrm{slide}}.
```

A prompt may describe a local feature:

```math
p
=
\text{tumor infiltrating lymphocytes}.
```

The image crop, slide embedding, and prompt must refer to the same object level.

## Language Bias

Text encoders inherit language frequencies:

```math
f_T(\text{common phrase})
\ne
f_T(\text{rare pathology phrase})
```

in ways not guaranteed to match visual pathology.

## Prompt Overfitting

Soft prompts can fit a small validation set:

```math
\eta^\star
=
\arg\min_\eta
\mathcal{L}_{\mathrm{val}}(\eta),
```

while learning prompt artifacts rather than robust concepts.

## Dense Summary

Prompt tuning is powerful because it changes the task interface cheaply. Its
failure mode is equally direct: the interface may ask the wrong question.
