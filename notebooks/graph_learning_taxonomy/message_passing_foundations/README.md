# Message Passing Foundations

This folder covers the first graph-learning portion.

The anchor question is:

```text
What exact operator updates each node from its neighbors?
```

## Files

- `01_message_passing_template.md`: general message-passing operator.
- `02_gcn_normalized_neighbor_averaging.md`: Kipf-Welling GCN as
  degree-normalized smoothing.
- `03_graphsage_sample_aggregate.md`: GraphSAGE as sampled local distribution
  estimation.
- `04_gat_masked_edge_attention.md`: GAT as data-dependent edge weighting.

## Shared Form

Every file can be placed into:

```math
h_v^{(\ell+1)}
=
\phi_{\theta}^{(\ell)}
\left(
h_v^{(\ell)},
\rho
\left(
\{\psi_{\theta}^{(\ell)}(h_v^{(\ell)},h_u^{(\ell)},e_{vu}):u\in\mathcal{A}(v)\}
\right)
\right).
```

The papers differ in:

```text
message:
    what is sent from u to v

aggregation:
    mean, normalized sum, sampled aggregate, attention-weighted sum

update:
    linear map, nonlinear map, concatenation, normalization

support:
    full neighborhood, sampled neighborhood, masked attention neighborhood
```
