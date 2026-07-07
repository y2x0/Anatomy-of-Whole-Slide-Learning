# Failure Modes

Pseudo-labels fail when model-generated supervision becomes self-reinforcing.

## Confirmation Bias

If a pseudo-label is wrong:

```math
\widehat Z_{ij}
\ne
Z_{ij},
```

and training increases confidence in it:

```math
P_{\theta_{t+1}}(\widehat Z_{ij}\mid h_{ij})
>
P_{\theta_t}(\widehat Z_{ij}\mid h_{ij}),
```

then the error becomes harder to correct.

## Threshold Bias

Confidence thresholding selects:

```math
M_{ij}
=
\mathbf{1}\{\max_c p_{ijc}\ge\tau\}.
```

This favors easy examples:

```math
P(M=1\mid \text{easy})
>
P(M=1\mid \text{hard}).
```

The model may never learn rare, ambiguous, or subtle morphology.

## Attention-Derived Label Drift

If pseudo-labels are derived from attention:

```math
\widehat Z
=
\Psi(a_\theta(H)),
```

then changes in attention change the training targets:

```math
a_{\theta_t}
\to
a_{\theta_{t+1}}
\quad
\Longrightarrow
\quad
\widehat Z_t
\to
\widehat Z_{t+1}.
```

This can create unstable training if selected patches flip across epochs.

## Teacher Error Transfer

Distillation transfers teacher structure:

```math
p_\theta
\approx
q_\phi.
```

If the teacher learned a shortcut, the student can inherit it even if the
student architecture is different.

## Pseudo-Bag Label Noise

Copying slide labels to pseudo-bags creates false positives:

```math
Y_i=1,
\qquad
Y_{im}^{\mathrm{true}}=0,
\qquad
\widehat Y_{im}=1.
```

This can train local models to treat background from positive slides as
positive.

## Dense Summary

Pseudo-labeling strengthens weak supervision by inventing missing labels:

```math
S
\to
\widehat U.
```

The core risk is that the invented labels become more influential than the
uncertainty that produced them.
