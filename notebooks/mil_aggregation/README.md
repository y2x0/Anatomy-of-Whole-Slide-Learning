# MIL Aggregation

This family asks:

```text
How do instances become a bag prediction?
```

The existing pooling, attention, graph, sequence, and state-space families
derive individual operators. This family is their MIL synthesis: it starts from
the latent bag problem and places each pathology paper by context operator,
readout operator, surviving statistic, and failure mode.

## Portions

```text
foundations/
    bag semantics, permutation symmetry, positive-instance assumptions,
    C/R/G/S decomposition, and surviving-statistic design
```

Later portions will derive hybrid MIL families from the mapped papers in the
private research map.

The mean/attention portion is now in `mean_attention_mil/`.

The transformer portion is now in `transformer_mil/`.

The graph MIL portion is now in `graph_mil/`. It derives graph-as-bag
equivariance, Patch-GCN, HEAT, WiKG, HACT, graph readouts, graph-specific
failure modes, and a C/R/G/S design matrix.

The state-space MIL portion is now in `state_space_mil/`. It derives ordered
bag symmetry, S4MIL, selective state spaces, MambaMIL, SR-Mamba reordering,
state-space readouts, and failure modes.

The hierarchical MIL portion is now in `hierarchical_mil/`. It derives
hierarchical bag objects, parent-map symmetry, region coarsening bottlenecks,
DTFD pseudo-bags, HIPT nested tokens, the HACT boundary, hierarchical
readouts, failure modes, and a C/R/G/S design matrix.
