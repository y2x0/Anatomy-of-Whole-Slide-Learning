# Failure Modes

## Under-Adaptation

If the ideal update lies outside the allowed set:

```math
\Delta^\star
\notin
\mathcal{A}_{\mathrm{PEFT}},
```

then optimization cannot reach the best task solution even with infinite data.

## Shortcut Drift

PEFT can latch onto shortcuts:

```math
\Delta_\eta
\text{ increases site/stain separability}
```

instead of disease morphology. Small updates are not automatically causal or
clinically meaningful.

## Layer Misplacement

If only late layers are adapted:

```math
\Delta W_\ell=0
\quad
\ell< L-k,
```

then early visual invariances remain fixed. This is a problem when the task
requires changing low-level morphology sensitivity.

## Rank Bottleneck

If multiple independent pathology axes must be adapted:

```math
a_1,\ldots,a_m
```

with:

```math
m>r,
```

a rank-$r$ update may collapse or trade off axes.

## Dense Summary

PEFT is attractive because it limits overfitting and cost. The price is a
specific adaptation bottleneck. The right question is whether the task fits
inside that bottleneck.
