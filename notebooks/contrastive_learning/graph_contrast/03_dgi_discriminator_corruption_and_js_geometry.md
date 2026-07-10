# DGI Discriminator, Corruption, And Jensen-Shannon Geometry

Primary anchor:

- Velickovic et al. "Deep Graph Infomax." ICLR 2019.
  https://arxiv.org/abs/1809.10341

## Positive And Negative Pair Laws

Let `P` be the distribution of uncorrupted local-summary pairs:

```math
P
=
p_{H,S}.
```

Let `Q_{\mathcal{C}}` be the pair law induced by corruption:

```math
Q_{\mathcal{C}}
=
p_{\widetilde H,S}^{\mathcal{C}}.
```

The population discriminator objective is:

```math
\mathcal{J}(D)
=
\mathbb{E}_{P}
\left[
\log D(H,S)
\right]
+
\mathbb{E}_{Q_{\mathcal{C}}}
\left[
\log
\left(
1-D(\widetilde H,S)
\right)
\right].
```

## Optimal Discriminator

Pointwise optimization gives:

```math
D^{\star}(h,s)
=
\frac{p(h,s)}
{p(h,s)+q_{\mathcal{C}}(h,s)}.
```

The optimal logit is a density ratio:

```math
\log
\frac{D^{\star}(h,s)}
{1-D^{\star}(h,s)}
=
\log
\frac{p(h,s)}
{q_{\mathcal{C}}(h,s)}.
```

Substitution yields:

```math
\max_D
\mathcal{J}(D)
=
-2\log 2
+
2
\mathrm{JS}
\left(
P
\|\|
Q_{\mathcal{C}}
\right).
```

DGI therefore separates the joint from a corruption-defined negative law. It
equals joint-versus-product-of-marginals discrimination only when the
corruption mechanism actually realizes that product law.

## Feature-Shuffling Corruption

In the transductive experiments, the paper preserves adjacency and row-shuffles
features:

```math
\widetilde A
=
A,
\qquad
\widetilde X
=
PX,
```

for random permutation matrix `P`.

The negative graph preserves:

```math
\text{degree sequence},
\quad
\text{edge set},
\quad
\text{multiset of feature rows},
```

while disrupting feature-to-position assignment.

Hence a successful discriminator must exploit dependence between topology and
where features occur. It need not encode every aspect of topology or every
feature coordinate.

## Impossible Corruption Limit

If corruption leaves the encoder distribution unchanged:

```math
P
=
Q_{\mathcal{C}},
```

then:

```math
D^{\star}
=
\frac{1}{2},
\qquad
\mathrm{JS}(P\|\|Q_{\mathcal{C}})
=
0.
```

No discriminator can learn from this relation.

## Trivial Corruption Limit

If positive and corrupted supports are disjoint:

```math
\mathrm{supp}(P)
\cap
\mathrm{supp}(Q_{\mathcal{C}})
=
\varnothing,
```

the discriminator saturates:

```math
D^{\star}=1
\quad P\text{-almost surely},
```

```math
D^{\star}=0
\quad Q_{\mathcal{C}}\text{-almost surely}.
```

The task can be solved through corruption artifacts without preserving useful
downstream graph semantics.

## Mean-Readout Collision

Two node-embedding multisets can satisfy:

```math
\left\{
h_1,ldots,h_N
\right\}
\ne
\left\{
h'_1,ldots,h'_{N'}
\right\}
```

but:

```math
\frac{1}{N}\sum_i h_i
=
\frac{1}{N'}\sum_j h'_j.
```

Then DGI's summary is identical after sigmoid. The local-global objective can
still distinguish local terms, but the global conditioning variable cannot
represent higher moments, multimodality, or spatial arrangement beyond what
has already entered each `h_i`.

## WSI Warning

For a WSI graph with many repeated benign patches, mean readout can converge
toward the dominant tissue mode:

```math
\frac{1}{N}
\sum_i h_i
\longrightarrow
\mathbb{E}_{p_{\mathrm{common}}}[H].
```

Rare tumor nodes contribute only in proportion to prevalence. DGI's global
signal can therefore underweight sparse diagnostic structure even when the
local encoder detects it.
