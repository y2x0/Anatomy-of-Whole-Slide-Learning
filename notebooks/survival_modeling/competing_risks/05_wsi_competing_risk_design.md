# WSI Competing-Risk Design

For a whole slide:

```math
H_i=\{h_{ij}\}_{j=1}^{n_i},
\qquad
z_i=\mathcal{R}(\mathcal{C}(H_i;G_i)).
```

Competing-risk heads decide how $z_i$ becomes event-type-specific risk.

## Shared Representation, Separate Cause Heads

The simplest design is:

```math
\eta_{ic}=w_c^\top z_i+b_c.
```

For cause-specific Cox:

```math
\lambda_c(t\mid z_i)
=
\lambda_{0c}(t)\exp(\eta_{ic}).
```

The slide encoder is shared. Each cause gets a different linear risk direction.

If $z_i$ is attention-pooled:

```math
z_i=\sum_j a_{ij}h_{ij},
```

then:

```math
\eta_{ic}
=
\sum_j a_{ij}w_c^\top h_{ij}.
```

The same attention weights are used for every cause. This can fail if recurrence
and non-cancer death depend on different regions.

## Cause-Specific Attention

Let:

```math
a_{ijc}
=
\operatorname*{softmax}_{j}(q_c^\top\phi(h_{ij})).
```

Then:

```math
z_{ic}=\sum_ja_{ijc}h_{ij},
\qquad
\eta_{ic}=w_c^\top z_{ic}.
```

This permits different event types to select different morphology.

Surviving statistic:

```text
one weighted first moment per cause
```

## Time-And-Cause-Specific Attention

For discrete competing risks:

```math
a_{ijkc}
=
\operatorname*{softmax}_{j}(q_{kc}^\top\phi(h_{ij})).
```

Then:

```math
z_{ikc}=\sum_ja_{ijkc}h_{ij}.
```

The event-time/cause logit is:

```math
u_{ikc}=w_{kc}^\top z_{ikc}+b_{kc}.
```

This is expressive but parameter-heavy. It answers:

```text
which morphology supports event type c at time bin k?
```

## Prototype Competing Risks

Let $p_{im}$ be prototype prevalence:

```math
p_{im}
=
\frac{1}{n_i}\sum_jq_{ijm}.
```

Cause-specific risk:

```math
\eta_{ic}
=
\sum_m\beta_{mc}p_{im}.
```

Discrete event-time/cause PMF:

```math
u_{ikc}
=
\sum_m\beta_{mkc}p_{im}.
```

This makes event types interpretable through morphology distributions.

## Graph Competing Risks

Let:

```math
\widetilde H_i=\operatorname{GNN}(H_i,G_i).
```

Cause readout:

```math
z_{ic}
=
\operatorname{READOUT}_{c}(\widetilde H_i).
```

Time-cause readout:

```math
z_{ikc}
=
\operatorname{READOUT}_{k,c}(\widetilde H_i).
```

Graph context is useful when event type depends on spatial arrangements such as
tumor-stroma interface, lymphocytic infiltration, necrotic neighborhoods, or
vascular invasion.

## Foundation Features

If patch embeddings come from a pathology foundation model:

```math
h_{ij}=E_{\operatorname{FM}}(x_{ij}),
```

then the competing-risk model learns:

```math
E_{\operatorname{FM}}
\to
\mathcal{C}
\to
\mathcal{R}_{c}
\to
\text{cause-specific risk}.
```

The foundation model may encode generic morphology, but the survival head still
defines which morphology becomes event-type-specific.

## Dense Design Table

```text
shared z, cause heads:
    cheap, but same slide statistic for all causes

cause-specific attention:
    interpretable event-specific regions

time-cause attention:
    most expressive, highest overfitting risk

prototype risks:
    interpretable distributional morphology

graph risks:
    spatial interactions can be cause-specific
```

## Key Mathematical Question

For every WSI competing-risk model, ask:

```math
H_i
\xrightarrow{\mathcal{C}}
\widetilde H_i
\xrightarrow{\mathcal{R}_{c}\ \text{or}\ \mathcal{R}_{k,c}}
z_{ic}\ \text{or}\ z_{ikc}
\xrightarrow{\mathcal{H}}
\lambda_{ic},F_{ic},p_{ikc}.
```

The critical design choice is where event type enters:

```text
only at the final head
in the readout
in the context operator
in the supervision/loss
```
