# Bidirectional Density Ratios And Nonidentifiability

Primary anchor:

- Radford et al. "Learning Transferable Visual Models From Natural Language
  Supervision." ICML 2021. https://arxiv.org/abs/2103.00020

## Image-To-Text Candidate Experiment

Fix an image `X=x`. Draw one positive text from the conditional distribution
and `B-1` distractors from the text marginal:

```math
T_{K}\sim p(t\mid x),
\qquad
T_j\sim p(t)
\quad
j\ne K.
```

With a uniform candidate index `K`, Bayes' rule gives:

```math
p
\left(
K=k
\mid
x,t_{1:B}
\right)
=
\frac{
p(t_k\mid x)/p(t_k)
}{
\sum_{j=1}^{B}p(t_j\mid x)/p(t_j)
}.
```

Therefore an unrestricted optimal row score satisfies:

```math
s^{\star}_{I\rightarrow T}(x,t)
=
\log
\frac{p(t\mid x)}{p(t)}
+
c(x)
=
\log
\frac{p(x,t)}{p(x)p(t)}
+
c(x).
```

## Text-To-Image Candidate Experiment

Reversing the proposal mechanism gives:

```math
s^{\star}_{T\rightarrow I}(x,t)
=
\log
\frac{p(x\mid t)}{p(x)}
+
d(t)
=
\log
\frac{p(x,t)}{p(x)p(t)}
+
d(t).
```

The common identifiable interaction is pointwise mutual information:

```math
\mathrm{PMI}(x,t)
=
\log
\frac{p(x,t)}{p(x)p(t)}.
```

Bidirectionality suppresses arbitrary row-only and column-only offsets because
one score matrix must support both conditionals.

## Restricted Bilinear Family

CLIP does not optimize over all measurable scores. It restricts:

```math
s_{\theta,\phi}(x,t)
=
\gamma
u_{\theta}(x)^{\top}v_{\phi}(t),
```

with unit vectors and finite dimension. The learned score is therefore a
low-rank spherical approximation to a density-ratio interaction, not the true
PMI in general.

For finite samples, the score matrix rank obeys:

```math
\mathrm{rank}(S)
\le
d.
```

If the empirical compatibility matrix requires rank greater than `d`, some
relations must be compressed or aliased.

## Pair Mechanism Changes The Ratio

Let `A=1` denote that a pair is admitted to the corpus. Training observes:

```math
p_{\mathrm{obs}}(x,t)
=
p(x,t\mid A=1).
```

The optimized interaction is then associated with:

```math
\log
\frac{
p(x,t\mid A=1)
}{
p(x\mid A=1)p(t\mid A=1)
},
```

not automatically with the target clinical population. Hashtag filtering,
caption splitting, model-based matching, or retrieval can all alter `A`.

## Semantic Nonidentifiability

Suppose a latent diagnostic state `Y` generates both modalities. Any invertible
relabeling `h` of that state gives:

```math
Y'=h(Y)
```

with the same observed joint distribution after corresponding changes in the
conditional decoders. The contrastive objective cannot identify a privileged
clinical meaning for coordinates of the embedding.

Even the inner-product geometry is invariant under a shared orthogonal map:

```math
u'_i=Qu_i,
\qquad
v'_j=Qv_j,
\qquad
Q^{\top}Q=I,
```

because:

```math
{u'_i}^{\top}v'_j
=
u_i^{\top}v_j.
```

Thus alignment identifies compatibility scores more directly than it
identifies human-interpretable axes.

## Failure Principle

If corpus membership is selected by an existing encoder, then the new model
learns a ratio under an encoder-shaped distribution. Better optimization of
that ratio does not remove the inherited selection geometry.
