# Label Noise Channel

Let $Y$ be the true label and $\widetilde Y$ the observed label.

A noisy-label channel is:

```math
T_{ab}
=
P(\widetilde Y=b\mid Y=a).
```

For $C$ classes:

```math
T
\in
[0,1]^{C\times C},
\qquad
\sum_{b=1}^{C}T_{ab}=1.
```

## Class-Conditional Noise

Class-conditional noise assumes:

```math
P(\widetilde Y=b\mid Y=a,H,G)
=
P(\widetilde Y=b\mid Y=a)
=
T_{ab}.
```

The observed class probability is:

```math
\widetilde p_b(x)
=
P(\widetilde Y=b\mid x)
=
\sum_{a=1}^{C}
T_{ab}p_a(x),
```

where:

```math
p_a(x)
=
P(Y=a\mid x).
```

In vector form:

```math
\widetilde p(x)
=
T^\top p(x).
```

## Binary Noise

For binary labels:

```math
\rho_+
=
P(\widetilde Y=0\mid Y=1),
\qquad
\rho_-
=
P(\widetilde Y=1\mid Y=0).
```

Then:

```math
P(\widetilde Y=1\mid x)
=
(1-\rho_+)P(Y=1\mid x)
+
\rho_-P(Y=0\mid x).
```

Thus:

```math
\widetilde p(x)
=
(1-\rho_+-\rho_-)p(x)+\rho_-.
```

If $\rho_++\rho_-<1$, the true posterior is:

```math
p(x)
=
\frac{\widetilde p(x)-\rho_-}
{1-\rho_+-\rho_-}.
```

## Instance-Dependent Noise

In WSI, noise is often instance-dependent:

```math
P(\widetilde Y\mid Y,H,G)
```

depends on morphology, report ambiguity, slide quality, or sampling.

Then a fixed transition matrix is misspecified:

```math
T(H,G)
\ne
T.
```

This is harder because the noise mechanism can share features with the true
signal.

## Report-Derived Labels

If labels come from reports, there may be an intermediate text variable:

```math
Y
\to
R
\to
\widetilde Y.
```

The report label may reflect:

```text
true diagnosis
uncertainty
negation
historical disease
sample adequacy
clinical context
template artifacts
```

Thus report-derived weak labels are noisy observations, not ground truth.

## C/R/G/S Placement

```text
G:
    noise can be associated with site, scanner, tissue area, or coordinate artifacts

C:
    high-capacity encoders can memorize noisy labels

R:
    pooling may select noise-correlated patches as witnesses

S:
    observed label is generated through T or T(H,G)
```

## Dense Summary

Noisy-label learning starts with:

```math
Y
\to
\widetilde Y.
```

If the channel is ignored, the model is trained to predict the corrupted label,
not the true label.
