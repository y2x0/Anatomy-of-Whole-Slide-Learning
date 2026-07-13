# Hybrid MIL

Hybrid MIL composes context, geometry, and readout operators from the other
families. Its mathematical question is not which operator is fashionable. It is
whether the composition changes the hypothesis class, the surviving statistic,
or only the parameterization.

A generic hybrid model is

```math
\widehat y_i
=
\mathcal H_{\omega}
\circ
\mathcal R_{\theta_R}
\circ
\mathcal C_{\theta_C}
\left(
H_i,G_i
\right).
```

The composition becomes nontrivial when C changes the support, ordering, or
geometry seen by R.

## Files

- `01_hybrid_composition_and_operator_noncommutativity.md`: composition,
  commutation, and the operator algebra of hybrid MIL.
- `02_graph_context_attention_readout_and_spatial_credit.md`: graph context
  followed by attention, with Patch-GCN and WiKG placements.
- `03_transformer_context_hierarchical_readout_and_long_range.md`: TransMIL,
  HIPT, and region-level transformer composition.
- `04_state_space_context_with_mil_readouts.md`: MambaMIL and SR-Mamba as
  ordered context operators with downstream readouts.
- `05_graph_hierarchy_attention_and_multiscale_context.md`: HACT, heterogeneous
  graph context, and attention at multiple levels.
- `06_hybrid_surviving_statistics_and_task_heads.md`: how hybrid composition
  interacts with classification and survival heads.
- `07_hybrid_mil_failure_modes_and_counterexamples.md`: noncommutativity,
  geometry mismatch, order sensitivity, and credit-assignment failures.
- `08_hybrid_mil_crgs_design_matrix_and_open_axes.md`: paper placements and
  unexplored C/R/G/S combinations.

The anchor papers are already mapped in the private research map: DSMIL,
TransMIL, Patch-GCN, HACT, WiKG, HIPT, DTFD-MIL, and MambaMIL. No source paper
is added by this portion.
