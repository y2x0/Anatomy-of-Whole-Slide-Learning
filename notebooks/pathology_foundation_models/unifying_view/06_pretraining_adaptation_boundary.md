# Pretraining and Adaptation Boundary

## 1. Frozen Encoder

```math
h=\Phi_{\phi_0}(X),
\qquad
\min_{\eta}\mathbb E\left[
\ell(\mathcal H_\eta(\mathcal R_\eta(h)),Y)
\right].
```

The pretraining contract remains fixed and only the downstream path moves.

## 2. Fine-Tuning

```math
\phi=\phi_0+\Delta_\eta,
\qquad
\min_{\eta}\mathbb E\left[
\ell(\mathcal H_\eta(\mathcal R_\eta(\Phi_{\phi}(X))),Y)
\right].
```

Fine-tuning can recover discarded task directions but can also erase useful
invariances under small cohorts.

## 3. Retrieval

Retrieval adapts the decision rule through memory without necessarily changing
the encoder:

```math
\widehat y=q\left(\Phi_{\phi_0}(X),\mathcal M_\eta\right).
```

The task enters stored examples, labels, or metric calibration rather than
weights.

## 4. Interpretation

Downstream interpretability should state whether it explains pretraining
features, adaptation parameters, retrieval evidence, or the final head.

