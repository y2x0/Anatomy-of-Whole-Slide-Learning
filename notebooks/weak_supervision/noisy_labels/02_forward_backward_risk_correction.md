# Forward And Backward Risk Correction

If the noise transition matrix is known or estimated, losses can be corrected.

Let the clean class posterior be:

```math
p_\theta(x)
\in
\Delta^{C-1}.
```

The noisy posterior is:

```math
\widetilde p_\theta(x)
=
T^\top p_\theta(x).
```

## Forward Correction

Forward correction trains the model through the noise channel:

```math
\mathcal{L}_{\mathrm{forward}}
=
-
\log
\left[
T^\top p_\theta(x)
\right]_{\widetilde y}.
```

This matches the observed noisy label distribution while keeping $p_\theta$ as
the clean posterior model.

## Backward Correction

Let $\ell(y,p)$ be the vector of losses for each clean class:

```math
\ell(p)
=
(\ell(1,p),\ldots,\ell(C,p))^\top.
```

Backward correction defines:

```math
\ell^{\leftarrow}(\widetilde y,p)
=
\left[
T^{-1}\ell(p)
\right]_{\widetilde y}.
```

Then:

```math
\mathbb{E}_{\widetilde Y\mid Y}
\left[
\ell^{\leftarrow}(\widetilde Y,p)
\right]
=
\ell(Y,p).
```

This is unbiased when $T$ is correct and invertible, but it can have high
variance and can produce negative loss values.

## Binary Corrected Posterior

For binary noise:

```math
\widetilde p
=
(1-\rho_+-\rho_-)p+\rho_-.
```

Thus:

```math
p
=
\frac{\widetilde p-\rho_-}{1-\rho_+-\rho_-}.
```

If $\rho_++\rho_-$ is close to $1$, correction is unstable because the
denominator is small.

## WSI Bag-Level Correction

For a bag model:

```math
p_i
=
P_\theta(Y_i=1\mid H_i,G_i).
```

The observed noisy label likelihood is:

```math
P(\widetilde Y_i=1\mid H_i,G_i)
=
(1-\rho_+)p_i+\rho_-(1-p_i).
```

The corrected negative log likelihood is:

```math
\mathcal{L}_i
=
-
\widetilde y_i
\log
\left(
(1-\rho_+)p_i+\rho_-(1-p_i)
\right)
-
(1-\widetilde y_i)
\log
\left(
\rho_+p_i+(1-\rho_-)(1-p_i)
\right).
```

## C/R/G/S Placement

```text
G:
    can be included in an instance-dependent T(H,G)

C:
    unchanged except gradients flow through corrected loss

R:
    produces clean-label bag probability p_i

S:
    observed label is modeled as noisy channel output
```

## Dense Summary

Risk correction inserts the noise channel into the loss:

```math
p_\theta(Y\mid X)
\to
P(\widetilde Y\mid X)
=
T^\top p_\theta(Y\mid X).
```

It is mathematically clean when $T$ is correct and fragile when $T$ is unknown or
instance-dependent.
