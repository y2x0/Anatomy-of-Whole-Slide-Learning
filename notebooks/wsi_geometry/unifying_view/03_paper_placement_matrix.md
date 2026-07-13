# Paper Placement Matrix

This matrix places WSI geometry anchors into the C/R/G/S language. It is not a
claim that a paper belongs to only one family. Many methods combine several
geometries.

## Matrix

| Anchor | Geometry G | \mathcal{C} | \mathcal{R} | S | Geometry Statistic |
|---|---|---|---|---|---|
| ABMIL / CLAM without coordinates | empty geometry | patchwise scoring or set attention | attention or class-specific pooling | slide label, pseudo-instance constraints for CLAM | morphology statistic without layout |
| TransMIL | sequence or position-aware token structure | transformer attention over WSI tokens | slide token or pooled embedding | slide label | global token interaction with positional structure |
| Patch-GCN | coordinate-derived patch graph | graph convolution / message passing | graph-level survival readout | survival loss | local spatial neighborhood context |
| HACT-Net | hierarchical cell-to-tissue graph | within-level and cross-level graph context | tissue/slide graph readout | classification loss | cell-to-tissue composition |
| HIPT | multiscale image hierarchy | local and global transformer context | hierarchical slide embedding | self-supervised and downstream task losses | nested field-of-view context |
| MambaMIL / SSM-MIL | ordered patch sequence | state-space or recurrent scan | sequence/slide readout | slide label | order-induced long-range context |
| WiKG | learned dynamic relation graph | knowledge-aware / dynamic graph attention | graph-level prediction | slide label | task-learned topology |
| PANTHER | prototype morphology geometry | assignment to prototype or mixture components | distribution/prototype summary | unsupervised prototype fitting plus task loss | morphology distribution, not necessarily physical layout |

## ABMIL / CLAM Without Coordinates

```text
G:
    empty

C:
    patch feature map, attention score, class-specific score

R:
    weighted first moment

S:
    slide label and optional pseudo-instance constraints
```

Geometry statement:

```math
f(\{(h_j,c_j)\}_j)
=
f(\{(h_j,c'_j)\}_j)
```

whenever the feature multiset is the same.

## TransMIL

TransMIL is geometry-relevant because transformer context acts over WSI tokens
with position structure. Full self-attention is complete-graph message passing
over tokens:

```math
\widetilde h_j
=
\sum_{k=1}^{n}
\alpha_{jk}v_k.
```

If positional information is injected, token interactions can depend on token
placement or sequence arrangement. Without meaningful position structure, the
model is closer to set attention.

## Patch-GCN

Patch-GCN-style models use WSI patches as graph nodes:

```math
V_i
=
\{1,\ldots,n_i\}.
```

Edges are derived from patch coordinates:

```math
(j,k)\in E_i
\quad\Longleftrightarrow\quad
k\in\mathrm{kNN}(c_j).
```

Message passing makes each patch state depend on a local tissue neighborhood:

```math
h_j^{(L)}
=
F_\theta(\{h_k:d_G(j,k)\le L\}).
```

For survival, the graph statistic is ultimately compressed into a risk score or
hazard representation.

## HACT-Net

HACT-style geometry has typed levels:

```math
V
=
V_{\mathrm{cell}}
\cup
V_{\mathrm{tissue}}.
```

Edges include:

```math
E
=
E_{\mathrm{cell-cell}}
\cup
E_{\mathrm{tissue-tissue}}
\cup
E_{\mathrm{cell-tissue}}.
```

The geometry statistic is not only adjacency but cross-scale containment:

```math
\text{cell morphology}
\to
\text{tissue region state}
\to
\text{slide prediction}.
```

## HIPT

HIPT-style methods use multiscale image hierarchy:

```math
\text{small patches}
\to
\text{regions}
\to
\text{slide}.
```

The geometry object is the field-of-view hierarchy:

```math
(h_v^{(\ell)},c_v^{(\ell)},\Delta_\ell).
```

The important bias is that local visual units compose larger visual units before
global slide reasoning.

## MambaMIL And State-Space MIL

State-space WSI methods impose an order:

```math
\sigma(1),\sigma(2),\ldots,\sigma(n).
```

The context path is:

```math
s_{t+1}
=
A_t s_t+B_t h_{\sigma(t)}.
```

The geometry statistic depends on the ordering. Different orderings induce
different notions of long-range and local context.

## WiKG And Learned Topology

Dynamic graph methods learn relations:

```math
A_{\theta,jk}
=
g_\theta(h_j,h_k,c_j,c_k).
```

The topology can connect patches that are not physically adjacent. This is
powerful when relation structure is semantic, but weakly identifiable when only
slide labels supervise edges.

## Dense Summary

```text
ABMIL/CLAM:
    geometry erased unless added externally

TransMIL:
    complete token interaction with positional structure

Patch-GCN:
    coordinate graph geometry

HACT:
    typed cell-to-tissue hierarchy

HIPT:
    multiscale image hierarchy

MambaMIL/SSM-MIL:
    sequence order as induced geometry

WiKG:
    learned dynamic topology

PANTHER:
    morphology geometry through prototype space
```
