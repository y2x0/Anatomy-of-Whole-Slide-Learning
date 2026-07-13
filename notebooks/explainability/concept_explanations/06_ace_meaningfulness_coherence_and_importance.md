# ACE Meaningfulness, Coherence, and Importance

ACE separates three desiderata that should not be collapsed into one score.

## 1. Meaningfulness

Meaningfulness asks whether humans assign a stable semantic description to a
concept's examples. For observer labels `L_{rk}` on cluster `k`, one simple
agreement functional is

```math
M_k
=\max_a\frac1R\sum_{r=1}^R
\mathbf 1\{L_{rk}=a\}.
```

High agreement depends on the observer population and vocabulary. It does not
show that the model uses the concept.

## 2. Coherence

Latent compactness can be measured by

```math
W_k
=\frac1{|\mathcal K_k|}
\sum_{z\in\mathcal K_k}\|z-\mu_k\|_2^2,
```

or by a human intrusion task: identify which candidate does not belong with
the others. Low latent variance is only a proxy for perceptual coherence because
the encoder metric may organize nuisance factors.

## 3. Importance

ACE uses TCAV to associate a discovered cluster with class sensitivity:

```math
I_{kc}
=\frac1{|X_c|}
\sum_{x\in X_c}
\mathbf 1\{
\nabla h_{l,c}(\phi_l(x))^{\top}v_k>0
\}.
```

This is sign-frequency sensitivity. The ACE desideratum informally invokes
necessity, but TCAV alone does not estimate necessity. A finite removal test
would instead require

```math
N_{kc}
=\mathbb E_{X\mid Y=c}
[F_c(X)-F_c(\mathcal D_k(X))],
```

where `D_k` is a valid concept-deletion operator.

## 4. Deletion and Insertion Ambiguity

Removing a visual concept requires selecting pixels, filling the removed
region, and preserving tissue plausibility. Adding one requires a generative
model or a compositing rule. Different operators induce different estimands:

```math
N_{kc}^{(a)}\ne N_{kc}^{(b)}
```

even for the same classifier and cluster.

## 5. Duplicate and Mixed Concepts

Overclustering produces duplicate concepts; underclustering produces mixed
concepts. Their TCAV scores are not additive because CAVs can be correlated and
the score discards magnitude. Concept discovery should therefore report cluster
stability, overlap, and pairwise CAV angles alongside rankings.

