# MoPaDi Linear Feature Guidance

MoPaDi combines a diffusion autoencoder feature space with task-specific
classifier guidance.

## 1. Linear Classifier

For normalized feature vector `z`, a binary or one-versus-rest logistic head
has logit

```math
s_c(z)=v_c^{\top}z+b_c.
```

In the paper's linear approach, the normalized classifier weight supplies the
feature-manipulation direction. Orient its sign toward the desired class:

```math
d_b=\frac{\sigma_b v_b}{\|v_b\|_2},
\qquad
\sigma_b\in\{-1,+1\}.
```

Choose `sigma_b` by verifying that motion increases the desired class score.
MoPaDi shifts features with increasing amplitude:

```math
z_{\mathrm{cf}}(\alpha)
=z+\alpha d_b.
```

For a multiclass pairwise margin that is represented by separate linear logits,
the steepest L2 direction is instead

```math
d_{a\to b}^{\mathrm{pair}}
=\frac{v_b-v_a}{\|v_b-v_a\|_2}.
```

This pairwise expression is a linear-head generalization, not an additional
formula reported by MoPaDi.

## 2. Exact Logit Motion

Along the pairwise margin direction,

```math
s_b(z+\alpha d)-s_a(z+\alpha d)
=s_b(z)-s_a(z)
+\alpha(v_b-v_a)^{\top}d.
```

The normalized weight difference maximizes first-order pairwise margin increase
per unit L2 latent displacement by Cauchy-Schwarz. For the paper's binary
guidance, the corresponding statement uses the oriented normalized classifier
weight.

## 3. Decode the Manipulation

After reversing feature normalization, the manipulated vector conditions the
DDIM decoder together with the original patch encoded into noise:

```math
x_{\mathrm{cf}}(\alpha)
=G_{\theta}(x_T(x),z_{\mathrm{cf}}(\alpha)).
```

This couples a mathematically simple classifier direction to a nonlinear image
trajectory.

## 4. What Is Explained

The generated sequence visualizes morphology that the MoPaDi feature extractor,
linear classifier, and decoder jointly associate with movement toward the
target class. It does not isolate a unique causal feature. Different generators
or equally predictive linear heads can produce different trajectories.

## 5. Boundary and Saturation

The analytic linear boundary may be crossed in feature space before the decoded
image visibly changes, or the image may change before a re-encoded feature
crosses. Counterfactual validity should be evaluated after decoding and
re-encoding, not inferred solely from the intended latent shift.
