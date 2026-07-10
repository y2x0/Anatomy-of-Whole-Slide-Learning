# Augmentation Orbits And Pathology Invariance

An augmentation pipeline defines which visible changes the representation is
asked to ignore. This is an invariance specification, even when it is written
as data preprocessing.

Primary anchors:

- SimCLR: https://arxiv.org/abs/2002.05709
- DSMIL: https://arxiv.org/abs/2011.08939
- CTransPath: https://pubmed.ncbi.nlm.nih.gov/35952419/

## Stochastic View Kernel

Let:

```math
V
\sim
\mathcal{A}
\left(
\cdot
\mid
X=x
\right).
```

The augmentation kernel can be represented as a distribution over
transformations:

```math
\mathcal{A}
\left(
v
\mid
x
\right)
=
\int
\delta_{T(x)}(v)
\,d\mu(T).
```

The contrastive positive joint is:

```math
p_{+}
\left(
v_1,v_2
\right)
=
\int
p_X(x)
\mathcal{A}
\left(
v_1
\mid
x
\right)
\mathcal{A}
\left(
v_2
\mid
x
\right)
\,dx.
```

## Deterministic Orbit Idealization

If the transformation family is a group `\mathfrak{T}`, the orbit of a
patch is:

```math
\mathcal{O}(x)
=
\left\{
T(x)
:
T\in\mathfrak{T}
\right\}.
```

Perfect invariance asks:

```math
f_{\theta}
\left(
T(x)
\right)
=
f_{\theta}(x),
\qquad
\forall T\in\mathfrak{T}.
```

Then the encoder factors through the quotient:

```math
\mathcal{X}
\xrightarrow{
q
}
\mathcal{X}/\mathfrak{T}
\xrightarrow{
\widetilde f_{\theta}
}
\mathcal{H},
```

so:

```math
f_{\theta}
=
\widetilde f_{\theta}
\circ
q.
```

Real augmentation libraries usually form neither a group nor exact orbits.
The quotient remains a useful local model of which distinctions are contracted.

## Pairwise Contraction

For a positive pair, a normalized squared alignment term is:

```math
\left\lVert
u_1-u_2
\right\rVert_2^2
=
2
-
2u_1^{\top}u_2.
```

Increasing positive cosine similarity contracts the expected within-source
scatter:

```math
\mathbb{E}
\left[
\left\lVert
U^{(1)}-U^{(2)}
\right\rVert_2^2
\mid
X
\right].
```

The transformation distribution decides which directions of image variation
are repeatedly contracted.

## Nuisance And Signal

Let image formation depend on biology `B` and nuisance `D`:

```math
X
\sim
p
\left(
x
\mid
B,D
\right).
```

An ideal nuisance augmentation varies `D` while preserving `B`:

```math
B
\left(
T(X)
\right)
=
B(X),
```

while:

```math
D
\left(
T(X)
\right)
\ne
D(X).
```

If the learned representation is invariant to that transformation:

```math
Z
\perp
D
\mid
B
```

is a desirable limiting interpretation.

It is not guaranteed. The contrastive loss observes transformations, not the
causal variables `B` and `D`.

## Destructive Invariance

Let `Y` be a downstream pathology target. A transformation is
label-preserving only if:

```math
p
\left(
Y
\mid
X=x
\right)
=
p
\left(
Y
\mid
X=T(x)
\right).
```

If this fails, exact invariance removes target information. By data processing:

```math
I
\left(
Y;
f_{\theta}(X)
\right)
\le
I
\left(
Y;
X
\right).
```

The inequality always holds for a deterministic encoder. Destructive
augmentation makes the gap larger by forcing target-relevant inputs into the
same representation neighborhood.

## Crop-Induced Partial Observability

Let a patch contain spatially indexed latent tissue:

```math
X
=
\left\{
X(r)
:
r\in\Omega
\right\}.
```

A crop selects a random support:

```math
\Omega_a
\subseteq
\Omega,
\qquad
V_a
=
X
\big\vert_{\Omega_a}.
```

For two views, the shared visible support is:

```math
\Omega_{12}
=
\Omega_1
\cap
\Omega_2.
```

If:

```math
\left|
\Omega_{12}
\right|
\ll
\left|
\Omega_1
\cup
\Omega_2
\right|,
```

then agreement can only be based on coarse or repeated attributes shared
across the views. A rare focus present in only one crop is pressured toward
irrelevance.

## A Minimal Pathology Counterexample

Let a source patch contain background state `N` and a small diagnostic
focus `R`:

```math
X
=
N
\cup
R.
```

Suppose:

```math
\Pr
\left(
R\subseteq\Omega_a
\right)
=
\rho,
\qquad
0<\rho<1.
```

The probability that exactly one of two independent crops contains the focus
is:

```math
2\rho
\left(
1-\rho
\right).
```

On those positive pairs, the objective asks a focus-present view and a
focus-absent view to agree. This is largest at:

```math
\rho
=
\frac{1}{2}.
```

The failure is caused by positive construction before any MIL aggregator is
applied.

## Color As Biology And Nuisance

Color transformations are often motivated by stain or scanner variation.
However, visible color can encode both:

```math
\text{technical nuisance}
```

and:

```math
\text{tissue composition or stain-dependent signal}.
```

The correct invariance is therefore conditional:

```math
f
\left(
T_D(x)
\right)
\approx
f(x)
```

for nuisance transformations `T_D`, but not necessarily for every
photometric transformation with the same pixel-level magnitude.

## Augmentation Strength As A Bias Parameter

Let `\alpha` index augmentation strength. The induced positive
conditional is:

```math
p_{+}^{(\alpha)}
\left(
v_1,v_2
\right).
```

Changing `\alpha` changes the statistical problem itself, not merely
optimization noise:

```math
s_{\alpha}^{\star}
\left(
v_1,v_2
\right)
=
\log
\frac{
p_{\alpha}
\left(
v_2
\mid
v_1
\right)
}{
\nu_{\alpha}(v_2)
}
+
c_{\alpha}(v_1).
```

There is no model-independent statement that stronger augmentation is better.

## C/R/G/S Placement

```text
\mathcal{G}:
    an augmentation-induced positive graph over patch views

\mathcal{C}:
    a patch encoder that contracts connected positive views

\mathcal{R}:
    the projected representation on which agreement is measured

\mathcal{S}:
    positive alignment plus the chosen anti-collapse or negative mechanism
```

## Failure Principle

An augmentation removes nuisance only when the variation it introduces is
independent of the target after conditioning on the preserved biological
state. Otherwise the objective learns invariance to a mixture of nuisance and
signal.
