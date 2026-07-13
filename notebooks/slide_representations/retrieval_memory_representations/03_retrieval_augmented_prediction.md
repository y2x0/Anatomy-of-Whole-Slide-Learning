# Retrieval-Augmented Prediction

Retrieval can be the whole predictor or a context source for a parametric model.

## Pure Nearest-Neighbor Prediction

For classification:

```math
\widehat p_i(y)
=
\sum_{k\in\mathcal{N}_K(i)}
w_{ik}\mathbf{1}\{y_k=y\}.
```

Prediction:

```math
\widehat y_i
=
\mathrm{argmax}_y
\widehat p_i(y).
```

This is nonparametric. The training data remain visible at inference time.

## Retrieval-Augmented Embedding

Let retrieved values be embeddings
```math
v_k
```
. A retrieved context vector is:

```math
r_i
=
\sum_{k\in\mathcal{N}_K(i)}w_{ik}v_k.
```

Then:

```math
\widehat y_i
=
\mathcal{H}_\theta([z_i;r_i]).
```

The representation is now:

```math
\widetilde z_i=[z_i;r_i].
```

This is parametric prediction with nonparametric memory context.

## Label-Distribution Context

If the memory stores labels:

```math
v_k=y_k,
```

then retrieval supplies a local label distribution:

```math
\widehat p_i
=
\sum_{k\in\mathcal{N}_K(i)}w_{ik}\delta_{y_k}.
```

A neural head can combine this with the query:

```math
\widehat y_i
=
\mathcal{H}_\theta(z_i,\widehat p_i).
```

This lets the model use both global decision boundaries and local case
analogies.

## Text And Report Context

If memory values are text embeddings:

```math
v_k=t_k,
```

then retrieval produces:

```math
r_i
=
\sum_{k\in\mathcal{N}_K(i)}w_{ik}t_k.
```

For vision-language models, prediction can compare the slide query to candidate
text prompts:

```math
p(y\mid S_i)
\propto
\exp
\left(
\frac{z_i^\top t_y}{\tau}
\right).
```

Retrieval can augment this with similar reports:

```math
p(y\mid S_i,\mathcal{M})
=
F_\theta(z_i,t_y,\{t_k:k\in\mathcal{N}_K(i)\}).
```

## Survival Retrieval

For survival tasks, memory values are:

```math
v_k=(T_k,\Delta_k).
```

A local Kaplan-Meier-style estimate can be computed over neighbors:

```math
\widehat S_i(t)
=
\prod_{\tau\le t}
\left(
1-
\frac{\sum_{k\in\mathcal{N}_K(i)}w_{ik}\mathbf{1}\{T_k=\tau,\Delta_k=1\}}
{\sum_{k\in\mathcal{N}_K(i)}w_{ik}\mathbf{1}\{T_k\ge\tau\}}
\right).
```

This makes risk a local neighborhood statistic instead of only a learned scalar.

## Dense Summary

```math
\boxed{
\widehat y_i
=
\mathcal{H}_\theta
\left(
z_i,
\sum_{k\in\mathcal{N}_K(i)}w_{ik}v_k
\right)
}
```

Retrieval-augmented prediction changes the slide representation from isolated
embedding to database-conditioned embedding.
