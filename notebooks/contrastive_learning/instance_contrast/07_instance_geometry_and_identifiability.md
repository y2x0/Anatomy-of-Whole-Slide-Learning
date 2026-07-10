# Instance Geometry And Identifiability

Instance contrast can identify a conditional density ratio or a candidate
classifier. It does not uniquely identify a biological coordinate system.

Primary anchors:

- CPC: https://arxiv.org/abs/1807.03748
- SimCLR: https://arxiv.org/abs/2002.05709
- DSMIL: https://arxiv.org/abs/2011.08939
- CTransPath: https://pubmed.ncbi.nlm.nih.gov/35952419/

## Critic-Level Identification

For a positive conditional `p(k\mid a)` and negative proposal
`\nu(k)`, the population-optimal unrestricted score is:

```math
s^{\star}(a,k)
=
\log
\frac{
p(k\mid a)
}{
\nu(k)
}
+
c(a).
```

The anchor-only shift `c(a)` is not identified because it cancels in a
candidate softmax:

```math
\frac{
\exp
\left(
s(a,k_j)+c(a)
\right)
}{
\sum_r
\exp
\left(
s(a,k_r)+c(a)
\right)
}
=
\frac{
\exp
\left(
s(a,k_j)
\right)
}{
\sum_r
\exp
\left(
s(a,k_r)
\right)
}.
```

Only relative candidate scores for a fixed anchor matter.

## Bilinear Restriction

A shared-encoder SimCLR-style critic, including the objective used for DSMIL
pretraining, restricts:

```math
s_{\theta}(a,k)
=
\frac{
u_{\theta}(a)^{\top}u_{\theta}(k)
}{
\tau
},
```

with:

```math
u_{\theta}(x)
\in
\mathbb{S}^{d-1}.
```

The learned Gram entries approximate the density-ratio score only within this
finite-dimensional, symmetric, bounded family.

Since:

```math
-1
\le
u(a)^{\top}u(k)
\le
1,
```

the logit range is:

```math
-
\frac{1}{\tau}
\le
s_{\theta}(a,k)
\le
\frac{1}{\tau}.
```

An unrestricted log density ratio need not be symmetric or bounded, so exact
identification is generally impossible.

Online-target methods can use different encoder parameters on the two sides,
so their score need not be symmetric:

```math
s_{\theta,\xi}(a,k)
=
\frac{
u_{\theta}(a)^{\top}u_{\xi}(k)
}{
\tau
},
```

while the unit-normalized dot product remains bounded.

## Orthogonal Non-Identifiability

Let:

```math
R^{\top}R
=
I.
```

The transformed embeddings:

```math
\widetilde u(x)
=
Ru(x)
```

preserve every dot product:

```math
\widetilde u(a)^{\top}\widetilde u(k)
=
u(a)^{\top}u(k).
```

Therefore a dot-product contrastive loss cannot identify an absolute axis
system. Individual embedding coordinates do not have stable biological
meaning without extra constraints.

## Projection-Head Non-Identifiability

Let the retained representation and contrastive representation be:

```math
h
=
f_{\theta}(x),
\qquad
z
=
g_{\phi}(h).
```

The contrastive objective constrains `z`. If two encoder-projector pairs
satisfy:

```math
g_{\phi}
\circ
f_{\theta}
=
\widetilde g_{\widetilde\phi}
\circ
\widetilde f_{\widetilde\theta},
```

they produce the same contrastive representation while their retained
`h` spaces can differ. Discarding the projector exposes a representation
not uniquely determined by the contrastive loss alone.

## Identity Does Not Identify Morphology

Let patch identity be `N` and latent morphology be `U`. The
positive experiment observes:

```math
N_1
=
N_2.
```

The desired semantic relation may be:

```math
U_1
=
U_2.
```

These are related only through the data-generating map:

```math
N
\longrightarrow
X_N
\longrightarrow
U(X_N).
```

Instance discrimination can preserve incidental identity information that
separates two patches with the same morphology:

```math
U_a
=
U_b,
\qquad
u(x_a)
\ne
u(x_b).
```

It can also contract two distinct morphologies if the augmentation or learned
feature family erases their difference.

## Proposal-Relative Geometry

The optimum depends on `\nu`. For two negative proposals:

```math
\nu_1(k),
\qquad
\nu_2(k),
```

the optimal scores differ by:

```math
s_1^{\star}(a,k)
-
s_2^{\star}(a,k)
=
\log
\frac{
\nu_2(k)
}{
\nu_1(k)
}
+
c_1(a)
-
c_2(a).
```

Changing from pan-cancer batches to organ-homogeneous batches changes the
identified contrast even when positive views are unchanged.

## Empirical Gram Geometry

For normalized embeddings:

```math
U
=
\begin{bmatrix}
u_1^{\top}\\
\vdots\\
u_N^{\top}
\end{bmatrix}
\in
\mathbb{R}^{N\times d},
```

the pairwise geometry is:

```math
G
=
UU^{\top}.
```

It satisfies:

```math
G
\succeq
0,
\qquad
\mathrm{diag}(G)
=
\mathbf{1},
\qquad
\mathrm{rank}(G)
\le
d.
```

The objective can constrain selected entries and softmax functions of
`G`, but many positive-semidefinite Gram matrices can have similar loss.

## A Repeated-Tissue Example

Suppose every slide contains normal stroma `S`, while only some contain a
rare tumor state `T`. Two stromal patches from different slides satisfy:

```math
U_a
=
U_b
=
S,
```

but are negatives under identity-only contrast.

If acquisition site is correlated with slide identity:

```math
D_a
\ne
D_b,
```

the easiest way to distinguish those negatives may be to retain site cues.
Instance discrimination has no internal reason to prefer:

```math
\text{tissue semantics}
```

over:

```math
\text{stable identity-correlated nuisance}.
```

## What SRCL Changes

SRCL adds a learned relation:

```math
u_a^{\top}u_b
\text{ is currently large}
\quad
\Longrightarrow
\quad
(a,b)
\in
E_{+}.
```

This can connect repeated tissue across identities. It still does not identify
why the similarity is large. The mined relation can represent morphology,
organ, stain, scanner, compression, or a mixture.

## What Survives

The identifiable object is closest to:

```math
\text{relative compatibility under the chosen positive and candidate experiment}.
```

The following are not identified by instance contrast alone:

```math
\text{absolute semantic axes},
\qquad
\text{a unique tissue partition},
\qquad
\text{slide-level relevance},
\qquad
\text{causal morphology}.
```

## Failure Principle

An embedding can solve same-patch candidate identification while being poorly
ordered for slide prediction. Pretraining success and downstream semantic
identification are different claims.
