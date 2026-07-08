# Graph Attention Failure Modes

Graph attention has both attention failure modes and graph failure modes.

## Wrong Edge Support

If the edge mask is:

```math
M_{uv}=0,
```

then:

```math
a_{uv}=0
```

regardless of score. Missing edges cannot be rescued by attention.

## Degree Bias

Neighborhood softmax normalizes over:

```math
|\mathcal{N}(u)|.
```

Nodes with many neighbors split mass more ways. Nodes with few neighbors can
receive concentrated messages even if none is strongly relevant.

## Attention Shortcut Edges

If graph construction includes artifact similarity, attention can amplify:

```math
\text{scanner artifact}
\to
\text{edge}
\to
\text{high message weight}.
```

The graph and attention can jointly stabilize a shortcut.

## Locality Trap

Spatial graphs often encode local neighborhoods:

```math
\|c_u-c_v\|\le r.
```

This helps local morphology but can miss global distributional facts. A tumor
pattern repeated across distant regions may need long-range aggregation.

## Interpretability Trap

High edge attention means:

```text
this message was used in this layer
```

It does not mean:

```text
this edge is biologically causal
```

The edge may simply reduce loss under the available labels.

## Dense Summary

Graph attention fails when:

```text
the graph is wrong
degree changes normalization
shortcuts define edges
locality blocks long-range evidence
edge weights are overinterpreted
```

