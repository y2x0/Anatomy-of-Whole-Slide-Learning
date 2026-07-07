# Supervised Contrastive MIL

Supervised contrastive learning uses labels to define positives.

For embeddings $u_i$ and labels $y_i$, positives for anchor $i$ are:

```math
\mathcal{P}(i)
=
\{p:p\ne i,\ y_p=y_i\}.
```

The supervised contrastive loss is:

```math
\mathcal{L}_{\mathrm{supcon}}
=
\sum_i
\frac{-1}{|\mathcal{P}(i)|}
\sum_{p\in\mathcal{P}(i)}
\log
\frac{
\exp(u_i^\top u_p/\tau)
}{
\sum_{a\ne i}
\exp(u_i^\top u_a/\tau)
}.
```

## WSI Slide-Level Use

If $u_i$ is a slide embedding:

```math
u_i
=
\mathcal{R}_\theta(\mathcal{C}_\theta(H_i,G_i)),
```

then same-label slides are pulled together.

This assumes:

```math
Y_i=Y_p
\quad\Longrightarrow\quad
\text{same relevant morphology}.
```

That is often only approximately true.

## Instance-Level MIL Ambiguity

If a slide is positive:

```math
Y_i=1,
```

only some patches may be positive. Pulling all instances from positive slides
together is wrong:

```math
h_{ij}
\sim
h_{pk}
\quad
\text{because}
\quad
Y_i=Y_p=1
```

does not imply:

```math
Z_{ij}=Z_{pk}=1.
```

A MIL-aware contrastive method should define positives through attention,
pseudo-labels, region labels, or slide-level embeddings rather than naively
using every patch.

## Class Imbalance

For imbalanced classes, supervised contrastive positives differ by class count:

```math
|\mathcal{P}(i)|
\approx
N_{y_i}-1.
```

Minority classes have fewer positives, and majority classes dominate sampled
denominators. Reweighting can use:

```math
w_i
\propto
\frac{1}{N_{y_i}}.
```

## Joint CE And SupCon

A common objective is:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda
\mathcal{L}_{\mathrm{supcon}}.
```

Cross entropy learns decision boundaries. SupCon shapes representation geometry.

## C/R/G/S Placement

```text
G:
    optional; positives may ignore or use spatial structure

C:
    representation is shaped by class-defined neighborhoods

R:
    slide or instance embeddings become contrastive objects

S:
    labels define positive and negative pairs
```

## Dense Summary

Supervised contrastive MIL is strong when:

```math
Y_i=Y_k
```

means the slides share meaningful morphology. It is weak or harmful when a class
contains multiple unrelated phenotypes.
