# Interpretability And Diagnostics

Attention weights are useful diagnostics, but they are not automatically
explanations.

This folder separates:

```text
weight:
    how a representation was constructed

evidence:
    how a feature contributed to the output

causality:
    what would change the output under intervention
```

Folder map:

```text
01_attention_is_not_explanation.md
    attention weights versus evidence and causal claims

02_effective_support_entropy_and_mass.md
    entropy, support, mass concentration, and diagnostics

03_gradient_attention_and_evidence.md
    combining weights with value gradients and logits

04_failure_modes.md
    explanation collapse, shortcut maps, and false localization
```

