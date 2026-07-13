# Information And ELBO View

Weak supervision can be understood as information loss.

Let
```math
U
```
 be the latent truth and
```math
S^{\mathrm{obs}}
```
the observed supervision:

```math
U
\to
S^{\mathrm{obs}}.
```

If
```math
S^{\mathrm{obs}}
```
 is generated from
```math
U
```
through an observation channel, then
by data processing:

```math
I(U;S^{\mathrm{obs}}\mid H,G)
\le
H(U\mid H,G).
```

Usually:

```math
I(U;S^{\mathrm{obs}}\mid H,G)
<
H(U\mid H,G),
```

because
```math
S^{\mathrm{obs}}
```
 is a compressed or corrupted view of
```math
U
```
.

## Missing Information

Define the missing information as:

```math
\mathcal{M}
=
H(U\mid S^{\mathrm{obs}},H,G).
```

If
```math
\mathcal{M}=0
```
, the observed supervision identifies the latent truth.

If:

```math
\mathcal{M}>0,
```

then multiple latent explanations remain possible after observing
```math
S^{\mathrm{obs}}
```
.

For bag labels:

```math
S^{\mathrm{obs}}
=
Y
=
\max_j Z_j.
```

The missing information is:

```math
H(Z\mid Y,H,G).
```

This is large when many instance-label configurations explain the same bag
label.

## Complete-Data Likelihood

If
```math
U
```
were observed, one could maximize:

```math
\log P_\theta(U\mid H,G).
```

But under weak supervision, the observed likelihood must include the observation
channel:

```math
\log P_{\theta,\alpha}(S^{\mathrm{obs}}\mid H,G)
=
\log
\sum_U
Q_\alpha(S^{\mathrm{obs}}\mid U,H,G)
P_\theta(U\mid H,G).
```

The sum over
```math
U
```
is where latent ambiguity enters.

## ELBO

For any variational distribution
```math
q(U)
```
:

```math
\log P_{\theta,\alpha}(S^{\mathrm{obs}}\mid H,G)
\ge
\mathbb{E}_{q(U)}
\left[
\log Q_\alpha(S^{\mathrm{obs}}\mid U,H,G)
+
\log P_\theta(U\mid H,G)
-
\log q(U)
\right].
```

The gap is:

```math
\mathrm{KL}
\left(
q(U)
\;\|\;
P_{\theta,\alpha}(U\mid S^{\mathrm{obs}},H,G)
\right).
```

Thus exact inference requires the posterior:

```math
P_{\theta,\alpha}(U\mid S^{\mathrm{obs}},H,G).
```

Weakly supervised WSI methods usually do not optimize this exact ELBO. The ELBO
view is a diagnostic analogy unless the method explicitly defines
```math
U
```
,
```math
Q
```
, and
a variational family
```math
q
```
.

## Method Families As Posterior Analogies

Mean or attention MIL:

```text
does not explicitly estimate U; learns a bag statistic predictive of observed supervision
```

Max MIL:

```math
q(U)
\approx
\delta_{\widehat j}
```

where
```math
\widehat j
```
is the selected witness.

Noisy-or MIL:

```math
q(Z_j=1)
\approx
P_\theta(Z_j=1\mid Y,H)
```

under conditional independence.

CLAM:

```math
q(U)
\approx
\delta_{\Psi(a,Y)}
```

on selected top/bottom attention instances, but this is only an analogy unless a
probabilistic model makes attention extremes a MAP estimate.

Teacher-student:

```math
q(U)
\approx
q_\phi(U\mid H,G)
```

where the teacher supplies a posterior-like target.

Contrastive learning:

```math
q(U_a\sim U_b)
```

is replaced by pair construction rules.

## Data Processing Failure

If:

```math
U
\to
S^{\mathrm{obs}}
\to
\theta
```

then the model cannot recover components of
```math
U
```
that leave no statistical trace
in
```math
S^{\mathrm{obs}}
```
 or
```math
H,G
```
.

For instance localization:

```math
I(Z_j;S^{\mathrm{obs}}\mid H,G)
```

may be small even when:

```math
I(Y;S^{\mathrm{obs}}\mid H,G)
```

is large.

This explains why slide classification can work while patch interpretation
fails.

## Sufficient Supervision

A supervision signal
```math
S^{\mathrm{obs}}
```
 exactly recovers a target
```math
T(U)
```
if:

```math
H(T(U)\mid S^{\mathrm{obs}},H,G)
=
0.
```

For prediction rather than exact recovery, the weaker useful statement is that
the Bayes risk using
```math
S^{\mathrm{obs}},H,G
```
equals the Bayes risk using the full
latent object
```math
U,H,G
```
 for the target functional
```math
T
```
.

Bag labels may be sufficient for slide classification:

```math
T(U)=Y.
```

They are usually not sufficient for instance truth:

```math
T(U)=Z_j.
```

## Dense Summary

Weak supervision is information compression:

```math
U
\to
S^{\mathrm{obs}}.
```

The central quantity is:

```math
H(U\mid S^{\mathrm{obs}},H,G).
```

If this conditional entropy is large, any confident latent explanation requires
assumptions beyond the observed supervision.
