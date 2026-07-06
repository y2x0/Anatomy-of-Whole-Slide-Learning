# Foundation Latent Representations

A foundation-latent representation treats the slide as an object in a pretrained
latent space.

The core question:

```text
What geometry did pretraining impose before the downstream task?
```

A foundation model can provide:

```text
patch latent space
slide latent space
text-aligned latent space
retrieval latent space
multimodal generative latent space
```

## Files

- `01_slide_as_latent_object.md`: pretrained representation spaces and frozen
  geometry.
- `02_frozen_patch_encoder_and_slide_encoder.md`: patch encoders versus
  slide-level encoders.
- `03_multimodal_alignment.md`: image-text and report-aligned slide spaces.
- `04_failure_modes.md`: pretraining bias, latent collapse, and prompt mismatch.

## Anchor Methods

UNI and Virchow are patch-level pathology foundation encoders. HIPT and
GigaPath introduce slide or multiscale slide encoders. CONCH, PRISM, and TITAN
make visual representation interact with text or report supervision.

The key distinction:

```text
using foundation features is not the same as having a foundation slide representation
```
