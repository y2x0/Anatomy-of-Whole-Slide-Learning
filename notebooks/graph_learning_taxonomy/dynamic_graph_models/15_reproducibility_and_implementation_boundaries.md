# Reproducibility And Implementation Boundaries

This note records the boundary between dynamic construction and layerwise rewiring, the operator-changing paper-code differences, and the final failure audit.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## 13. The Graph Is Constructed Once

The released WiKG forward pass builds one graph from the pre-context patch
states:

```math
A^{(0)}
=
\Gamma_K(H^{(0)}).
```

It then performs one knowledge-aware aggregation and fusion. It does not
recompute:

```math
A^{(1)}
=
\Gamma_K(H^{(1)}).
```

Therefore the model cannot revise neighbor identities after seeing its own
contextualized node states. Calling the method dynamic refers to slide- and
parameter-dependent construction, not iterative graph rewiring.

## 14. Paper-Code Differences Change The Operator

Three distinctions materially affect reproduction.

Similarity normalization:

```text
printed equation:
    raw dot product divided by a raw dot-product sum

paper algorithm and code:
    top-k raw scaled logits, then softmax over selected logits
```

Triplet compatibility:

```text
paper equation:
    matching-coordinate dot product

released einsum:
    product of the two coordinate sums
```

Readout:

```text
paper:
    mean or max examples

model class default:
    attention

released training script:
    mean
```

These are not cosmetic notation choices. They change edge states, local
attention, and the final slide statistic.

## Failure-Mode Matrix

| Mechanism | Mathematical bottleneck | Observable audit |
|---|---|---|
| hard top-k | discontinuous support at rank boundaries | top-k margin and edge stability |
| fixed K | equal inbound source budget for every target | sweep K and inspect weak selected scores |
| dense discovery | all-pairs score matrix | peak memory versus patch count |
| no coordinates | learned relation is not physical adjacency | edge-length distribution in slide space |
| encoder dependence | nuisance similarity becomes topology | stain/scanner perturbation stability |
| two softmax stages | concentration or dilution | entropy of omega and pi separately |
| local weighted mean | neighborhoods collide after aggregation | compare higher-order neighborhood moments |
| global mean readout | graph fields collide after pooling | readout ablation and distributional probes |
| weak identification | many graphs fit slide labels | seed, fold, and encoder edge agreement |
| paper-code divergence | different executed operators | report exact equation and commit used |

## C/R/G/S Failure Placement

```text
\mathcal{G} failures:
    unstable top-k, fixed K, self-edges, shortcut topology, quadratic discovery

\mathcal{C} failures:
    triplet-kernel discrepancy, attention collapse, weighted-mean collision

\mathcal{R} failures:
    mean/max/attention compression and configuration ambiguity

\mathcal{S} failures:
    slide labels do not identify biological edges
```

## Dense Summary

WiKG's strongest inductive claim is also its main risk:

```math
\text{predictive head-tail compatibility}
\quad\Longrightarrow\quad
\text{useful directed WSI relation}.
```

The implication can hold for prediction without holding for physical or
biological interpretation. Dynamic topology should therefore be evaluated as
both a predictive mechanism and a claimed relational object.
