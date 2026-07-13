# Objective, Geometry, and Evaluation Matrix

| Evaluation | What it probes | What it does not prove |
|---|---|---|
| linear probe | accessible label boundary | nonlinear or spatial sufficiency |
| MIL readout | patch-to-slide aggregation compatibility | pretraining mechanism |
| retrieval | local neighbor coherence | causal or calibrated diagnosis |
| survival | risk ordering and possibly calibration | diagnostic morphology completeness |
| segmentation | local and boundary information | long-range slide reasoning |
| zero-shot text | image-language alignment | expert semantic truth |
| external cohort | transfer under shift | all unseen domains |

## Geometry Mismatch

Let two metrics be `d_1` and `d_2`. A model can have strong linear separability
under one and weak retrieval under the other:

```math
\mathrm{Acc}_{\mathrm{probe}}(d_1)
\not\Rightarrow
\mathrm{mAP}_{\mathrm{retrieval}}(d_1).
```

Benchmarks should match the readout geometry the model claims to support.

