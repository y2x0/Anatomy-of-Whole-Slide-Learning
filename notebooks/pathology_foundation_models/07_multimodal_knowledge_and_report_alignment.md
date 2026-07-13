# Multimodal Knowledge and Report Alignment

## 1. Knowledge-Conditioned Representation

Let image, report, gene, or pathway modalities be encoded as

```math
H^{(m)}=f_m(X^{(m)}).
```

A multimodal foundation objective aligns or fuses them:

```math
z=\mathcal F(H^{(1)},\ldots,H^{(M)}).
```

TITAN, KEEP, CONCH, and related models differ in how the auxiliary knowledge
enters the representation geometry.

## 2. Report Alignment

For paired slide and report representations, a contrastive term pulls matched
pairs together and separates mismatched pairs. A matched representation can be
written

```math
z_{\mathrm{slide}}=f_{\mathrm{slide}}(X),
\qquad
z_{\mathrm{text}}=f_{\mathrm{text}}(R).
```

The alignment preserves report-visible information, not necessarily every
microscopic feature a pathologist could use.

## 3. Knowledge Bias

If the auxiliary modality contains a shortcut `U`, then

```math
R\leftarrow U\rightarrow Y
```

can make the multimodal representation predictive without learning the desired
visual mechanism. Missing reports, templated language, and institution-specific
phrasing alter the learned geometry.

## 4. Interpretability Boundary

Cross-modal attention or retrieval can reveal a semantic relation. It does not
automatically establish that the image region caused the report concept or the
downstream prediction.

