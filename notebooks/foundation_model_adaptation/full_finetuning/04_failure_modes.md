# Failure Modes

## Compute-Limited Sampling

Fine-tuning may use only a subset of patches:

```math
\mathcal{S}_i
\subset
\{1,\ldots,n_i\}.
```

The gradient estimates:

```math
\sum_{j\in\mathcal{S}_i}
\frac{\partial\mathcal{L}}{\partial h_{ij}}
\frac{\partial h_{ij}}{\partial\phi},
```

not the full-slide gradient. Bad sampling can adapt the encoder to the wrong
tissue.

## Shortcut Adaptation

If slide labels correlate with site:

```math
Y
\not\perp
\text{site},
```

full fine-tuning can encode site more strongly than frozen features did.

## Attention Feedback

When attention selects wrong patches early:

```math
\mathcal{T}^{+}
\cap
\{j:Z_j=1\}
=
\varnothing,
```

end-to-end gradients can make the encoder increase the separability of those
wrong patches.

## Dense Summary

Full fine-tuning is the most expressive adaptation regime. It is also the
easiest way for weak labels to corrupt the representation itself.
