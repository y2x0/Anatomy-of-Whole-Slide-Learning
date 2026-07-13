# Minimal Reporting Rule and Paper Placement

## 1. Minimal Reporting Fields

Every explanation result should report:

```text
slide object;
encoder and representation space;
context operator;
readout operator;
target output and sign;
explanation quantity;
intervention semantics;
reference or baseline;
spatial resolution;
faithfulness test;
plausibility test;
stability test;
causal scope;
```

Missing fields are not cosmetic omissions: they prevent the explanation from
being mathematically identified.

## 2. Paper Placement

```math
\boxed{
\text{paper}
\mapsto
(\text{object},\text{context},\text{readout},\text{target},
\text{quantity},\text{intervention},\text{validation})
}
```

Examples:

```text
MCAT -> pathology/omics bags, co-attention, survival risk, routing map
HEAT -> heterogeneous graph, message passing, node-removal loss, localization
ProtoMIL -> sparse concept field, attention sum, linear risk/class credit
C2MIL -> graph MIL, semantic/topological intervention, causal subgraph
```

## 3. Final Principle

The mathematically honest explanation is not the most visually persuasive
one. It is the one whose displayed object, target, operator, and validation
claim all refer to the same forward computation.

