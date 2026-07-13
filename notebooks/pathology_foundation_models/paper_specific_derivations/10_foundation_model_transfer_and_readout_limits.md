# Foundation Model Transfer and Readout Limits

## 1. Frozen Transfer

With frozen encoder `Phi`, downstream learning is restricted to

```math
\widehat y=\mathcal H_\eta
\left(\mathcal R_\eta(\Phi(X))\right).
```

Performance measures accessibility of the task in the retained representation
and chosen readout class.

## 2. Three Transfer Failures

```text
missing statistic: the encoder discarded the task signal;
readout bottleneck: the signal exists but the head cannot extract it;
distribution shift: the signal geometry changes across cohorts.
```

Fine-tuning can address the second and sometimes the first, but may damage
pretrained invariances on small pathology cohorts.

## 3. Evaluation Matrix

```math
\mathcal E
=\{
\text{linear probe},
\text{MIL readout},
\text{retrieval},
\text{survival},
\text{external shift}
\}.
```

A foundation model claim should state which member of `E` was evaluated.

## 4. Explainability Consequence

An interpretable downstream head explains the composition of the frozen
representation, not necessarily how the foundation model learned its features.
Pretraining objective, encoder attention, and downstream attention are separate
explanation targets.

