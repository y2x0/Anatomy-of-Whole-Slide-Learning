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

The limiting case makes the degree effect explicit. If all neighborhood scores
are equal for node `u`, then:

```math
a_{uv}
=
\frac{1}{|\mathcal{N}(u)|}
\qquad
v\in\mathcal{N}(u).
```

Attention then computes a uniform neighborhood mean; the individual edge mass
depends on degree even though the aggregate remains normalized.

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

For a purely local `L`-layer message-passing stack with no global shortcuts:

```math
\mathrm{dist}_G(u,v)>L
\quad\Longrightarrow\quad
\frac{\partial h_u^{(L)}}{\partial h_v^{(0)}}
=
0.
```

This is a support limitation, not merely a weaker weight. Hierarchical pooling,
long-range edges, or a global readout can change the statement by adding a new
communication path.

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
