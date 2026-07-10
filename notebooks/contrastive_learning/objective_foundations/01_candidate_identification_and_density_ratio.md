# Candidate Identification And Density Ratios

InfoNCE is a multiclass identification problem. Its population optimum is
determined by the distributions used to place one positive among sampled
candidates.

Primary source:

- van den Oord, Li, and Vinyals. "Representation Learning with Contrastive
  Predictive Coding." 2018. https://arxiv.org/abs/1807.03748

## Generative Experiment

Draw an anchor:

```math
A
\sim
p_A.
```

Choose the positive index:

```math
I
\sim
\mathrm{Uniform}
\{1,\ldots,N\}.
```

Conditional on `A=a` and `I=i`, generate:

```math
K_i
\sim
p_{K\mid A}(\cdot\mid a),
```

```math
K_j
\overset{\mathrm{iid}}{\sim}
\nu_K,
\qquad
j\ne i.
```

The conditional candidate density is:

```math
p(k_{1:N}\mid a,I=i)
=
p_{K\mid A}(k_i\mid a)
\prod_{j\ne i}
\nu_K(k_j).
```

## Bayes Posterior

Bayes' rule gives:

```math
p(I=i\mid a,k_{1:N})
=
\frac{
p_{K\mid A}(k_i\mid a)
\prod_{j\ne i}\nu_K(k_j)
}{
\sum_{r=1}^{N}
p_{K\mid A}(k_r\mid a)
\prod_{j\ne r}\nu_K(k_j)
}.
```

Multiply numerator and denominator by:

```math
\left(
\prod_{j=1}^{N}\nu_K(k_j)
\right)^{-1}.
```

Then:

```math
p(I=i\mid a,k_{1:N})
=
\frac{
r_{\nu}(a,k_i)
}{
\sum_{r=1}^{N}
r_{\nu}(a,k_r)
},
```

where the proposal-relative density ratio is:

```math
r_{\nu}(a,k)
=
\frac{
p_{K\mid A}(k\mid a)
}{
\nu_K(k)
}.
```

## Optimal Critic

Let a score-based classifier be:

```math
q_{\psi}(I=i\mid a,k_{1:N})
=
\frac{
\exp(f_{\psi}(a,k_i))
}{
\sum_{r=1}^{N}
\exp(f_{\psi}(a,k_r))
}.
```

The population cross-entropy is minimized when the model posterior equals the
Bayes posterior. Therefore an optimal critic satisfies:

```math
f^{\star}(a,k)
=
\log
p_{K\mid A}(k\mid a)
-
\log
\nu_K(k)
+
c(a),
```

where `c(a)` is arbitrary because adding the same anchor-dependent constant to
every candidate logit leaves softmax unchanged.

Equivalently:

```math
\exp(f^{\star}(a,k))
\propto
r_{\nu}(a,k).
```

This is the density-ratio result derived in CPC.

## Cross-Entropy Decomposition

Let:

```math
\mathcal{D}_N
=
(A,K_1,\ldots,K_N).
```

For any candidate classifier:

```math
\mathcal{L}_N(q_{\psi})
=
H(I\mid\mathcal{D}_N)
+
\mathbb{E}_{\mathcal{D}_N}
\left[
\mathrm{KL}
\left(
p(I\mid\mathcal{D}_N)
\,
\Vert
\,
q_{\psi}(I\mid\mathcal{D}_N)
\right)
\right].
```

Hence:

```math
\mathcal{L}_N(q_{\psi})
\ge
H(I\mid\mathcal{D}_N),
```

with equality if and only if the modeled posterior equals the Bayes posterior
almost surely.

## Marginal Negatives And Pointwise Mutual Information

If negatives come from the true key marginal:

```math
\nu_K
=
p_K,
```

then:

```math
f^{\star}(a,k)
=
\log
\frac{p_{AK}(a,k)}
{p_A(a)p_K(k)}
+
c(a).
```

The nonconstant term is pointwise mutual information between anchor and key.

If instead:

```math
\nu_K
\ne
p_K,
```

the optimal critic estimates a different ratio. The loss does not correct this
automatically.

## Candidate Count Does Not Change The Population Ratio

The Bayes ratio:

```math
r_{\nu}(a,k)
```

does not depend on `N`. Candidate count changes:

```text
classification difficulty
finite-sample gradient competition
the largest attainable MI lower bound
the variance and composition of the denominator
```

but not the population form of the optimal pairwise critic under the stated
generative experiment.

## Asymmetry

The density ratio is conditional:

```math
r_{\nu}(a,k)
=
\frac{p(k\mid a)}{\nu(k)}.
```

It need not satisfy:

```math
r_{\nu}(a,k)
=
r_{\nu}(k,a).
```

Symmetric image-view losses explicitly average two directional identification
problems. Symmetry is an objective design choice, not a consequence of
InfoNCE.

## C/R/G/S Placement

```text
\mathcal{G}:
    proposal nu_K and positive conditional p(K|A)

\mathcal{C}:
    anchor and key encoders

\mathcal{R}:
    projected object embedding entering f_psi

\mathcal{S}:
    generated positive index I and candidate cross-entropy
```

## Dense Summary

InfoNCE learns similarity relative to a proposal:

```math
\text{learned compatibility}
\approx
\log
\frac{
\text{positive conditional}
}{
\text{negative proposal}
}.
```

Changing the sampler changes the statistical target even when the loss formula
is unchanged.
