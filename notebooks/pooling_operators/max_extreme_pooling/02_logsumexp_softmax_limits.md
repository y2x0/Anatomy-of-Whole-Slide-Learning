# Log-Sum-Exp And Softmax Limits

Hard max is nondifferentiable at ties and sends gradient only to the winning
instance. Smooth max operators interpolate between mean-like and max-like
behavior.

## Log-Sum-Exp Pooling

For instance scores $s_1,\ldots,s_n$, define:

```math
\mathrm{LSE}_{\beta}(s)
=
\frac{1}{\beta}
\log
\left(
\sum_{j=1}^{n}\exp(\beta s_j)
\right),
\qquad
\beta>0.
```

This approximates max:

```math
\lim_{\beta\to\infty}
\mathrm{LSE}_{\beta}(s)
=
\max_j s_j.
```

The standard bound is:

```math
\max_j s_j
\le
\mathrm{LSE}_{\beta}(s)
\le
\max_j s_j+\frac{\log n}{\beta}.
```

Large $\beta$ makes the operator close to max. Small $\beta$ spreads influence
across more instances.

## Stabilized Form

Let:

```math
m=\max_j s_j.
```

Then:

```math
\mathrm{LSE}_{\beta}(s)
=
m
+
\frac{1}{\beta}
\log
\left(
\sum_j\exp(\beta(s_j-m))
\right).
```

This avoids numerical overflow and makes the approximation error explicit:

```math
0
\le
\mathrm{LSE}_{\beta}(s)-m
\le
\frac{\log n}{\beta}.
```

## Gradient Is Softmax

The derivative is:

```math
\frac{\partial \mathrm{LSE}_{\beta}(s)}
{\partial s_j}
=
\frac{\exp(\beta s_j)}
{\sum_\ell\exp(\beta s_\ell)}
=
a_j.
```

So log-sum-exp pooling sends gradient according to a softmax distribution:

```math
a_j
=
\mathrm{softmax}_{j}(\beta s).
```

As $\beta\to\infty$:

```math
a_j
\to
\mathbf{1}\{j=j^\star\}.
```

As $\beta$ decreases, more instances receive gradient.

## Softmax-Weighted Score Pooling

Another smooth extreme statistic is:

```math
r_\beta
=
\sum_j a_j s_j,
\qquad
a_j
=
\frac{\exp(\beta s_j)}
{\sum_\ell\exp(\beta s_\ell)}.
```

This also approaches max:

```math
\lim_{\beta\to\infty}r_\beta
=
\max_j s_j.
```

But it is not the same as log-sum-exp. Log-sum-exp includes an entropy term:

```math
\mathrm{LSE}_{\beta}(s)
=
\sum_j a_j s_j
+
\frac{1}{\beta}
H(a),
```

where:

```math
H(a)
=
-\sum_j a_j\log a_j.
```

Thus log-sum-exp rewards both high score and spread when $\beta$ is finite.

## Generalized Mean

For nonnegative scores $s_j\ge0$, the generalized mean is:

```math
M_p(s)
=
\left(
\frac{1}{n}
\sum_j s_j^p
\right)^{1/p}.
```

Limits:

```math
\lim_{p\to1}M_p(s)
=
\frac{1}{n}\sum_j s_j,
```

```math
\lim_{p\to\infty}M_p(s)
=
\max_j s_j.
```

The parameter $p$ controls how strongly the readout emphasizes large values.

## Temperature As Inductive Bias

The temperature or sharpness parameter determines the effective number of
instances:

```math
n_{\mathrm{eff}}
=
\frac{1}{\sum_j a_j^2}.
```

If $a$ is uniform:

```math
n_{\mathrm{eff}}=n.
```

If $a$ is one-hot:

```math
n_{\mathrm{eff}}=1.
```

So smooth max pooling chooses a point on the spectrum:

```text
diffuse evidence averaging <-> sparse winner-take-all evidence
```

## Dense Summary

```math
\boxed{
\mathrm{LSE}_{\beta}(s)
\to
\max_j s_j
\quad
\text{as}
\quad
\beta\to\infty
}
```

Smooth max operators trade off optimization stability against the inductive bias
that one instance should dominate the bag prediction.
