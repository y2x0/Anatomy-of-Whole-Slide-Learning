# Hierarchical MIL

This is the hierarchical MIL portion.

```math
\mathcal H_i
=
\left(
\{V_i^{(\ell)}\}_{\ell=0}^{L},
\{\pi_i^{(\ell)}\}_{\ell=0}^{L-1}
\right).
```
# Hierarchical MIL

This portion asks how a slide-level predictor composes evidence across nested
units rather than aggregating one flat patch bag.

The central object is a rooted forest of units:

```math
\mathcal H_i
=
\left(
\{V_i^{(\ell)}\}_{\ell=0}^{L},
\{\pi_i^{(\ell)}\}_{\ell=0}^{L-1},
\{H_i^{(\ell)}\}_{\ell=0}^{L}
\right).
```

The level-zero units may be patches or cells. A parent map
\pi_i^{(\ell)} assigns each level-\ell unit to one level-\ell+1 unit. A
hierarchical MIL model is therefore a composition of

```math
\text{fine context}
\longrightarrow
\text{child-to-parent coarsening}
\longrightarrow
\text{coarse context}
\longrightarrow
\text{slide readout}.
```

The hierarchy changes the hypothesis class. A flat set model is invariant to
all permutations of instances. A fixed hierarchy is invariant only to
permutations that preserve the parent map, unless the parent map itself is
learned equivariantly.

## Files

- `01_hierarchical_bag_object_and_parent_maps.md`: formal forests, levels,
  parent maps, and hierarchical symmetry.
- `02_region_coarsening_and_information_bottlenecks.md`: assignment operators,
  rank, nullspaces, and what coarsening destroys.
- `03_dtfd_pseudo_bags_and_two_tier_distillation.md`: paper-faithful
  pseudo-bags, first-tier screening, second-tier fusion, and distillation.
- `04_hipt_nested_tokens_and_multiscale_readout.md`: HIPT's nested visual
  tokens and the boundary between pretraining and downstream aggregation.
- `05_hact_and_hierarchical_graph_mil_boundary.md`: HACT, HIPT, and DTFD as
  different answers to what the hierarchy means.
- `06_hierarchical_readouts_and_surviving_statistics.md`: level-wise readouts
  and the statistics that survive each scale.
- `07_hierarchical_mil_failure_modes_and_counterexamples.md`: partition,
  bottleneck, selection, leakage, and complexity failures.
- `08_hierarchical_mil_crgs_design_matrix.md`: a unified C/R/G/S placement and
  design rules for new hierarchical MIL models.

## Anchor papers

The derivations use the mapped sources already tracked in the private research
map:

- DTFD-MIL, Zhang et al., CVPR 2022:
  https://arxiv.org/abs/2203.12081
- HIPT, Chen et al., CVPR 2022:
  https://arxiv.org/abs/2206.02647
- HACT-Net and its hierarchical graph formulation:
  https://arxiv.org/abs/2007.00584

The notes keep three meanings separate:

- hierarchical visual pretraining, as in HIPT;
- hierarchical pseudo-bag distillation, as in DTFD-MIL;
- hierarchical graph message passing and assignment, as in HACT.

A hierarchy is not automatically a better MIL aggregator. It imposes a
partition, a sequence of bottlenecks, and a scale at which evidence is allowed
to interact.
