# Positive-Unlabeled Bag View

MIL bag labels create a positive-unlabeled problem at the instance level.

Negative bags identify negative instances:

```math
Y_i=0
\quad\Longrightarrow\quad
Z_{ij}=0
\quad
\forall j.
```

Positive bags contain a mixture:

```math
Y_i=1
\quad\Longrightarrow\quad
Z_{ij}
\text{ may be }0\text{ or }1.
```

Thus instances from positive bags are unlabeled, not automatically positive.

## Instance Distributions

Let:

```math
P_0(h)
=
P(h\mid Z=0),
\qquad
P_1(h)
=
P(h\mid Z=1).
```

Negative-bag instances are sampled from $P_0$ under the standard assumption.

Positive-bag instances are sampled from a mixture:

```math
P(h\mid Y=1)
=
\pi P_1(h)
+
(1-\pi)P_0(h),
```

where:

```math
\pi
=
P(Z=1\mid Y=1).
```

The class prior $\pi$ is generally unknown and slide-dependent.

## Naive Instance Training Bias

If every patch from a positive bag is labeled positive, the model trains on:

```math
\widetilde Z_{ij}
=
Y_i.
```

Then the positive instance distribution is:

```math
P(h\mid \widetilde Z=1)
=
P(h\mid Y=1)
=
\pi P_1(h)+(1-\pi)P_0(h).
```

This teaches the classifier to separate positive-bag background from negative
background if those differ by cohort, tissue source, stain, or sampling.

## PU Risk Sketch

The true instance risk is:

```math
R(g)
=
\pi\mathbb{E}_{P_1}[\ell(g(h),1)]
+
(1-\pi)\mathbb{E}_{P_0}[\ell(g(h),0)].
```

Using the positive-bag mixture:

```math
\mathbb{E}_{P(h\mid Y=1)}[\ell(g(h),1)]
=
\pi\mathbb{E}_{P_1}[\ell(g(h),1)]
+
(1-\pi)\mathbb{E}_{P_0}[\ell(g(h),1)].
```

Without correcting for the negative component in positive bags, the instance
classifier is biased.

## Bag-Level Safe Statement

Bag-level training avoids falsely labeling all positive-bag patches as positive.
It supervises:

```math
Y_i
=
\Gamma(Z_i)
```

directly. The price is that $Z_i$ remains latent.

## C/R/G/S Placement

```text
G:
    may help distinguish true positives from positive-bag background

C:
    learns instance embeddings whose separability is weakly constrained

R:
    maps the latent mixture to a bag prediction

S:
    negative bags are clean instance negatives; positive bags are unlabeled mixtures
```

## Dense Summary

At the instance level:

```text
negative bag patches:
    negative

positive bag patches:
    unlabeled mixture
```

That is why naive patch labeling from slide labels is statistically wrong.
