# Set Transformer Pooling

Set Transformer pooling uses learned query or seed vectors to extract
statistics from permutation-equivariant set states.

The core question:

```text
Which learned queries should read out the set?
```

The canonical readout is pooling by multihead attention:

```math
Z_i
=
\mathrm{MHA}(S,\widetilde H_i,\widetilde H_i),
```

where $S$ is a learned set of seed vectors.

## C/R/G/S Placement

```text
G:
    complete learned relation graph unless external geometry is added

C:
    self-attention blocks or inducing-point attention before pooling

R:
    pooling by multihead attention with learned seed queries

S:
    task loss shaping seed queries and set interactions
```

Surviving statistic:

```math
Z_i
=
\{z_{i1},\ldots,z_{iK}\},
```

a learned multi-query set summary.

## Files

- `01_pma_seed_queries.md`: pooling by multihead attention and seed statistics.
- `02_sab_isab_context_before_pooling.md`: set attention blocks and inducing
  points.
- `03_set_transformer_vs_attention_mil.md`: ABMIL versus PMA.
- `04_failure_modes.md`: quadratic cost, seed collapse, and interaction loss.

## Anchor Paper

- Lee et al. "Set Transformer." ICML 2019.
  https://arxiv.org/abs/1810.00825

