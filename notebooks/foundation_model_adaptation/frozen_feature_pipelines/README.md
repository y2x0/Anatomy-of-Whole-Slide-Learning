# Frozen Feature Pipelines

Frozen-feature adaptation treats the foundation model as a fixed map:

```math
h_{ij}
=
f_{\phi_0}(x_{ij}),
\qquad
\nabla_{\phi_0}\mathcal{L}
=
0.
```

Task information enters only after feature extraction:

```math
H_i
\xrightarrow{\mathcal{C}_\eta,\mathcal{R}_\eta}
z_i
\xrightarrow{\mathcal{H}_\eta}
\widehat y_i.
```

## Files

- `01_patch_encoder_as_fixed_map.md`: frozen patch encoders and downstream MIL.
- `02_slide_encoder_as_fixed_map.md`: frozen slide encoders and direct probing.
- `03_probe_vs_aggregator_adaptation.md`: head-only versus readout adaptation.
- `04_failure_modes.md`: frozen geometry mismatch, cohort shift, and invisible task axes.

## C/R/G/S Placement

```text
G:
    coordinates or slide-token geometry may be supplied after frozen encoding

C:
    task-specific context is trainable, but the foundation encoder is fixed

R:
    MIL, graph, transformer, prototype, or distribution readout adapts to labels

S:
    downstream labels shape only post-encoder parameters
```

## Dense Summary

Frozen features are strongest when the target is linearly or readout-separably
encoded in the pretrained geometry. They fail when the needed pathology
direction was never represented by $f_{\phi_0}$.
