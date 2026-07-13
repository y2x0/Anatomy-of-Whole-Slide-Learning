# Transformer Attention Failure Modes

Transformer attention is expressive but not automatically appropriate for WSI
scale.

## Quadratic Cost

Full self-attention costs:

```math
O(n^2d).
```

For large WSI bags, this forces approximations:

```text
patch sampling
token pruning
window attention
hierarchical attention
kernel or low-rank attention
state-space replacement
```

Each approximation changes the operator and the failure mode.

The approximation must be placed mathematically, not just named. For example:

```text
window attention:
    changes support

kernel attention:
    changes normalization and value aggregation

low-rank attention:
    changes the rank of the attention operator

token pruning:
    changes the source set before attention
```

If an efficient method replaces the exact attention matrix `A` by
`\widetilde A` while keeping the value matrix `V` fixed, the immediate output
error obeys:

```math
\|AV-\widetilde A V\|_F
\le
\|A-\widetilde A\|_2
\|V\|_F.
```

This is an operator-approximation statement, not a guarantee of equal task
performance. The value norm, later nonlinear layers, and the location of the
approximation in a deep stack determine how that error propagates.

For WSI, an "efficient" attention method only reduces cost if the expensive
dense score matrix is avoided, approximated, or factorized before full
materialization.

## Dense Denominator Dilution

For each query:

```math
a_{uv}
=
\frac{\exp(s_{uv})}
{\sum_r\exp(s_{ur})}.
```

When many irrelevant patches exist, relevant messages must compete with large
denominators. Attention can dilute rare signals even before readout.

## CLS Bottleneck

If one CLS token is the slide statistic:

```math
z
=
h_{\mathrm{cls}}^{(L)},
```

then a giant slide is compressed to one vector. The context may be rich, but the
surviving statistic is still a bottleneck.

## Geometry Omission

Without positional structure:

```math
\mathrm{Attn}(PH)
=
P\mathrm{Attn}(H).
```

The model cannot distinguish rearranged tissue layouts except through patch
features. This is a dangerous assumption for spatial pathology tasks.

## Spurious Long-Range Links

Full attention allows any patch to attend to any patch. This can learn useful
global context, but it can also connect artifacts, staining patterns, or batch
effects across the slide.

## Dense Summary

Transformer attention fails when:

```text
quadratic cost forces lossy approximations
rare evidence is diluted
one readout token bottlenecks the slide
geometry is omitted or misencoded
long-range attention learns shortcuts
```
