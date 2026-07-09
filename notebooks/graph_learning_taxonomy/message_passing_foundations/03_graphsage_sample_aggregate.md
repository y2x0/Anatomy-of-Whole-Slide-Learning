# GraphSAGE As Sample And Aggregate

GraphSAGE was designed to avoid learning only transductive node embeddings tied
to one fixed training graph. It changes the graph-learning question from:

```text
learn an embedding for every node in a fixed graph
```

to:

```text
learn a function that generates node embeddings from features and neighborhoods
```

That distinction matters for WSI because test slides are unseen graphs. It does
not mean GraphSAGE is the only inductive GNN; shared-parameter GCN and GAT
layers can also be applied to new graphs when node features and graph structure
are available. GraphSAGE makes the sample-and-aggregate inductive recipe
explicit.

## Mean GraphSAGE Layer

For node `v`, sample a neighborhood:

```math
S_{\ell}(v)
\subseteq
\mathcal{A}(v).
```

The sampled neighbor aggregate is:

```math
\bar h_{\mathcal{A}(v)}^{(\ell)}
=
\frac{1}{|S_{\ell}(v)|}
\sum_{u\in S_{\ell}(v)}
h_u^{(\ell)}.
```

The update is:

```math
h_v^{(\ell+1)}
=
\sigma
\left(
W^{(\ell)}
\left[
h_v^{(\ell)}
\Vert
\bar h_{\mathcal{A}(v)}^{(\ell)}
\right]
\right).
```

Then normalization is often applied:

```math
h_v^{(\ell+1)}
\leftarrow
\frac{h_v^{(\ell+1)}}
{\|h_v^{(\ell+1)}\|_2}.
```

## Message-Passing Placement

GraphSAGE with mean aggregation uses:

```math
m_{vu}^{(\ell)}
=
h_u^{(\ell)},
\qquad
u\in S_{\ell}(v).
```

Aggregation:

```math
\rho
=
\mathrm{mean}.
```

Update:

```math
U_{\theta}^{(\ell)}
\left(
h_v,\bar m_v
\right)
=
\sigma
\left(
W^{(\ell)}
[h_v\Vert\bar m_v]
\right).
```

The self-state and neighbor summary are kept as separate channels until the
linear map mixes them.

## Sampling As An Estimator

If:

```math
S_{\ell}(v)
\subseteq
\mathcal{A}(v)
```

is sampled uniformly, the sampled mean estimates the full neighbor mean:

```math
\mu_v^{(\ell)}
=
\frac{1}{|\mathcal{A}(v)|}
\sum_{u\in\mathcal{A}(v)}
h_u^{(\ell)}.
```

The estimator is:

```math
\widehat\mu_v^{(\ell)}
=
\frac{1}{|S_{\ell}(v)|}
\sum_{u\in S_{\ell}(v)}
h_u^{(\ell)}.
```

Under uniform sampling:

```math
\mathbb{E}
\left[
\widehat\mu_v^{(\ell)}
\right]
=
\mu_v^{(\ell)}.
```

But the variance depends on local heterogeneity:

```math
\mathrm{Var}
\left(
\widehat\mu_v^{(\ell)}
\right)
\propto
\frac{1}{|S_{\ell}(v)|}
\mathrm{Var}_{u\in\mathcal{A}(v)}
\left(
h_u^{(\ell)}
\right).
```

Thus sampling is not just engineering. It changes which local tissue
distribution the model actually sees per step.

## Inductive Meaning

GraphSAGE learns:

```math
f_{\theta}
\left(
h_v,
\{h_u:u\in\mathcal{A}(v)\}
\right)
```

instead of a table of node embeddings.

Therefore it can be applied to a new graph:

```math
G_{\mathrm{test}}
\ne
G_{\mathrm{train}}.
```

For WSI, this is useful because:

```text
each patient slide:
    new graph

each tissue mask:
    new node set

each patch sampling:
    new neighborhood structure
```

## What Survives

The node state after one mean GraphSAGE layer preserves:

```text
self feature
sampled local first moment
```

After `L` layers, the node stores a sampled, nested neighborhood statistic:

```math
h_v^{(L)}
=
F_{\theta}^{(L)}
\left(
h_v^{(0)},
\widehat\mu_v^{(0)},
\widehat\mu_{\mathcal{A}(v)}^{(1)},
\ldots
\right).
```

The exact statistic is random because sampled neighborhoods are random.

## WSI Failure Mode

Suppose the relevant signal is an interface pattern:

```text
rare tumor-immune boundary patches
```

If the sampled neighborhood misses those boundary nodes:

```math
S_{\ell}(v)
\cap
V_{\mathrm{interface}}
=
\varnothing,
```

then the update cannot include that evidence at layer `ell`.

Sampling reduces computation, but it can also increase variance precisely in
heterogeneous neighborhoods where pathology context matters most.

## Dense Summary

GraphSAGE mean aggregation is:

```math
\boxed{
h_v'
=
\sigma
\left(
W
\left[
h_v
\Vert
\frac{1}{|S(v)|}
\sum_{u\in S(v)}
h_u
\right]
\right)
}
```

Its inductive bias is sampled local distribution estimation. The surviving
neighbor statistic is a sampled first moment, not the full graph neighborhood.
