# Mean After Context

The formula:

```math
z_i
=
\frac{1}{n_i}
\sum_j u_{ij}
```

does not tell us enough. We must ask what $u_{ij}$ already contains.

## Raw Mean

If no context is used:

```math
u_{ij}=h_{ij}.
```

Then:

```math
z_i
=
\frac{1}{n_i}
\sum_j h_{ij}.
```

This is the first moment of raw patch embeddings:

```math
z_i
=
\mathbb{E}_{\mu_i}[h].
```

No explicit pairwise, spatial, or regional relation survives unless it is
already encoded in $h_{ij}$.

## Mean After Instance Transform

If:

```math
u_{ij}
=
\phi_\theta(h_{ij}),
```

then:

```math
z_i
=
\mathbb{E}_{\mu_i}[\phi_\theta(h)].
```

This can preserve nonlinear instance statistics, but there is still no
cross-instance interaction before pooling.

## Mean After Graph Context

A graph context operator produces:

```math
u_{ij}
=
\mathrm{GNN}_\theta(H_i,A_i)_j.
```

Mean pooling gives:

```math
z_i
=
\frac{1}{n_i}
\sum_j
\mathrm{GNN}_\theta(H_i,A_i)_j.
```

This is a first moment of contextualized node states. The graph can encode local
spatial neighborhoods into each $u_{ij}$ before averaging.

What survives:

```text
average graph-contextualized morphology
```

What can still be lost:

```text
topology not embedded into node states before the final mean
```

If two slides have the same multiset of final node states:

```math
\{\!\{u_{ij}\}\!\}
=
\{\!\{u_{kj}\}\!\},
```

then mean readout cannot distinguish them, even if the original graphs differ.

## Mean After Full Attention

A self-attention layer gives:

```math
u_{ij}
=
\sum_{\ell=1}^{n_i}
a_{j\ell}v(h_{i\ell}).
```

Then:

```math
z_i
=
\frac{1}{n_i}
\sum_j
\sum_{\ell=1}^{n_i}
a_{j\ell}v(h_{i\ell}).
```

Rearrange:

```math
z_i
=
\sum_{\ell=1}^{n_i}
\left(
\frac{1}{n_i}
\sum_j a_{j\ell}
\right)
v(h_{i\ell}).
```

Thus mean after attention becomes a weighted average of value features, where the
effective weight for instance $\ell$ is:

```math
\bar a_{\ell}
=
\frac{1}{n_i}
\sum_j a_{j\ell}.
```

Even if final readout is a mean, attention has already changed which instances
carry influence into the statistic.

## Mean After Sequence Context

For a state-space or recurrent sequence model:

```math
u_{it}
=
\mathcal{C}_{\text{seq}}
(h_{i\sigma(1)},\ldots,h_{i\sigma(n_i)})_t.
```

Mean pooling gives:

```math
z_i
=
\frac{1}{n_i}
\sum_t u_{it}.
```

The readout is order-invariant over the output states, but the states themselves
are order-dependent.

Changing $\sigma$ can change $u_{it}$:

```math
u_{it}^{\sigma}
\ne
u_{it}^{\sigma'}.
```

So:

```math
\frac{1}{n_i}\sum_t u_{it}^{\sigma}
\ne
\frac{1}{n_i}\sum_t u_{it}^{\sigma'}.
```

The final mean does not erase the sequence assumption.

## Context-Then-Mean Template

The general form is:

```math
z_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\mathcal{C}_\theta(\mathcal{X}_i)_j.
```

The statistic is a first moment, but over the image of the context operator:

```math
z_i
=
\mathbb{E}_{j}
[
\mathcal{C}_\theta(\mathcal{X}_i)_j
].
```

This distinction matters:

```text
same R, different C, different surviving information
```

## Dense Summary

```math
\boxed{
\text{mean readout}
=
\text{first moment of whatever states context produced}
}
```

Mean pooling after context can preserve interactions only if the context
operator already encoded those interactions into instance states.
