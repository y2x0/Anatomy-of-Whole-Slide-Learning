# Information And ELBO View

Weak supervision can be understood as information loss.

Let $U$ be the latent truth and $S$ the observed supervision:

```math
U
\to
S.
```

If $S$ is generated from $U$ through an observation channel, then by data
processing:

```math
I(U;S\mid H,G)
\le
H(U\mid H,G).
```

Usually:

```math
I(U;S\mid H,G)
<
H(U\mid H,G),
```

because $S$ is a compressed or corrupted view of $U$.

## Missing Information

Define the missing information as:

```math
\mathcal{M}
=
H(U\mid S,H,G).
```

If $\mathcal{M}=0$, the observed supervision identifies the latent truth.

If:

```math
\mathcal{M}>0,
```

then multiple latent explanations remain possible after observing $S$.

For bag labels:

```math
S
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

If $U$ were observed, one could maximize:

```math
\log P_\theta(U\mid H,G).
```

But under weak supervision, the observed likelihood is:

```math
\log P_\theta(S\mid H,G)
=
\log
\sum_U
P_\theta(S,U\mid H,G).
```

The sum over $U$ is where latent ambiguity enters.

## ELBO

For any variational distribution $q(U)$:

```math
\log P_\theta(S\mid H,G)
\ge
\mathbb{E}_{q(U)}
\left[
\log P_\theta(S,U\mid H,G)
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
P_\theta(U\mid S,H,G)
\right).
```

Thus exact inference requires the posterior:

```math
P_\theta(U\mid S,H,G).
```

Weakly supervised WSI methods usually approximate this posterior implicitly.

## Method Families As Posterior Approximations

Mean or attention MIL:

```text
does not explicitly estimate U; learns a bag statistic predictive of S
```

Max MIL:

```math
q(U)
\approx
\delta_{\widehat j}
```

where $\widehat j$ is the selected witness.

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

on selected top/bottom attention instances.

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
S
\to
\theta
```

then the model cannot recover components of $U$ that leave no statistical trace
in $S$ or $H,G$.

For instance localization:

```math
I(Z_j;S\mid H,G)
```

may be small even when:

```math
I(Y;S\mid H,G)
```

is large.

This explains why slide classification can work while patch interpretation
fails.

## Sufficient Supervision

A supervision signal $S$ is sufficient for a target $T(U)$ if:

```math
T(U)
\perp
U
\mid
S,H,G.
```

Equivalently, observing $S,H,G$ preserves all information about $T(U)$ needed
for prediction.

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
S.
```

The central quantity is:

```math
H(U\mid S,H,G).
```

If this conditional entropy is large, any confident latent explanation requires
assumptions beyond the observed supervision.
