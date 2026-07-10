# Graph Design Matrix And Failure Principle

This note closes graph learning with a compact comparison matrix, a design-lattice view of expressivity, and a common construction-context-readout-head failure decomposition.

The four anchor WSI graph families differ most sharply in what determines an
edge and whether slide supervision can change that edge.

Specific anchors:

```text
Patch-GCN:
    Chen et al., Whole Slide Images are 2D Point Clouds, MICCAI 2021.
    https://arxiv.org/abs/2107.13048

HACT:
    Pati et al., HACT-Net, 2020.
    https://arxiv.org/abs/2007.00584
    Pati et al., Hierarchical graph representations in digital pathology,
    Medical Image Analysis 2022.
    https://arxiv.org/abs/2102.11057

HEAT:
    Chan et al., Histopathology Whole Slide Image Analysis With Heterogeneous
    Graph Representation Learning, CVPR 2023.
    https://arxiv.org/abs/2307.04189

WiKG:
    Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
    Histopathology Whole Slide Image Analysis, CVPR 2024.
    https://arxiv.org/abs/2403.07719
```

## Comparison Matrix

| Axis | Patch-GCN | HACT | HEAT | WiKG |
|---|---|---|---|---|
| node ontology | patch | cell and tissue region | patch with pseudo-type | patch with head and tail roles |
| support source | coordinates | entity geometry and containment | encoder feature kNN | learned head-tail logits |
| directed relation | not central in paper text | assignment is directed across levels | message orientation on feature support | explicit tail-to-head direction |
| relation attributes | coordinate support | entity level and assignment | pseudo-types plus continuous edge attribute | endpoint-derived relation vector |
| context | DeepGCN-style graph operator | cell GNN, transfer, tissue GNN | HEAT | knowledge-aware attention |
| readout | global attention | tissue sum or sum-concat | pseudo-type pooling then graph pooling | released script uses mean |
| supervision | survival | region classification | WSI classification/staging/typing | WSI classification/staging |
| can task loss reshape support? | no | no | no in presented pipeline | yes, across top-k boundaries |
| main bottleneck | finite coordinate support | segmentation and hard assignment | pseudo-type and feature-kNN errors | weakly identified unstable topology |

## Expressivity Does Not Form A Total Order

WiKG has a more flexible support map:

```math
A_{\theta}(H)
```

but does not dominate the biological ontology of HACT or HEAT. HACT exposes
cell containment explicitly. HEAT exposes pseudo-type and edge attributes.
Patch-GCN preserves physical locality directly.

Conversely, those explicit structures cannot freely learn arbitrary distant
directed relations without changing their graph constructor.

The methods form a design lattice, not a ranking:

```text
physical prior strength
task-adaptive support
entity ontology
relation typing
readout richness
```

Moving along one axis does not guarantee improvement along another.

## Unified Failure Principle

Every graph family can fail at four distinct places:

```math
\underbrace{X\to\mathcal{G}}_{\text{construction error}}
\to
\underbrace{\mathcal{C}(H;\mathcal{G})}_{\text{context error}}
\to
\underbrace{\mathcal{R}(\widetilde H)}_{\text{readout collision}}
\to
\underbrace{\mathcal{H}(z)}_{\text{task mismatch}}.
```

For the four anchors:

```text
Patch-GCN:
    coordinate graph can omit distant morphology

HACT:
    segmentation or assignment can route cells incorrectly

HEAT:
    pseudo-types or feature support can encode the wrong ontology

WiKG:
    slide labels can reward a predictive but non-biological topology
```

## Final C/R/G/S View

```math
\boxed{
\widehat y
=
\mathcal{H}
\circ
\mathcal{R}
\circ
\mathcal{C}
\left(
H;
\mathcal{G}_{\theta_G}(X)
\right)
}
```

Patch-GCN, HACT, and HEAT primarily differ in the externally constructed
object supplied as `G`. WiKG makes `theta_G` task-trainable under supervision
`S`; the inference-time graph remains a function of the slide and learned
parameters, not the unknown test label.

The graph-learning question is therefore not only:

```text
How does information move through a graph?
```

It is also:

```text
Who decided that the edge existed, and can the task loss change that decision?
```
