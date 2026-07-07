# Prompt Tuning

Prompt tuning adapts a foundation model by changing input tokens or text
queries rather than most model weights.

For a vision-language model:

```math
s_c(x)
=
\frac{
f_I(x)^\top f_T(p_c)
}{\tau},
```

where $p_c$ is the prompt for class $c$.

## Files

- `01_text_prompt_as_classifier.md`: zero-shot and prompt-defined class scores.
- `02_soft_prompts_and_context_vectors.md`: learned context vectors.
- `03_visual_prompts_and_token_prefixes.md`: visual/prefix adaptation.
- `04_prompt_ensembles_and_semantics.md`: prompt averaging and semantic variance.
- `05_failure_modes.md`: prompt mismatch, label wording, and local-global mismatch.

## C/R/G/S Placement

```text
G:
    inherited visual-text geometry

C:
    prompt changes the conditioning context

R:
    prompt score or prompt-conditioned embedding becomes the readout

S:
    task information enters through words, soft prompt vectors, or prefix tokens
```

## Dense Summary

Prompt tuning asks whether the task can be solved by asking the pretrained
model the right question.
