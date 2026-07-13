# Graph MIL C/R/G/S Design Matrix

## 1. Unified Forward Map

For slide `i`, write the graph MIL predictor as:

```math
\mathcal{G}_{i,\theta}
=
\Gamma_{\theta}
\left(
H_i^{(0)},C_i,\text{optional labels}
\right),
```

```math
\widetilde H_i
=
\mathcal{C}_{\theta}
\left(
H_i^{(0)},\mathcal{G}_{i,\theta}
\right),
```

```math
z_i
=
\mathcal{R}_{\theta}
\left(
\widetilde H_i,\mathcal{G}_{i,\theta}
\right),
```

```math
\widehat y_i
=
\mathcal{H}_{\theta}(z_i).
```

The graph can be fixed from coordinates, constructed from feature similarity,
learned by top-k scores, or supplied as a hierarchy. These are different
hypothesis classes even when the final readout is mean pooling.

## 2. Paper-Faithful Matrix

| Method | `G`: graph object | `C`: context operator | `R`: graph readout | Surviving statistic | Main failure |
| --- | --- | --- | --- | --- | --- |
| Patch-GCN | coordinate kNN patch graph | DeepGCN-style local softmax, residual and dense layers | global scalar attention | weighted first moment of contextualized multiscale nodes | wrong spatial support, local smoothing |
| HEAT | feature kNN patch graph with pseudo-types and Pearson edge attributes | node-type-specific, edge-attribute transformer | PL-Pool then graph readout | type-conditioned summaries | teacher/type error, feature geometry shortcut |
| WiKG | learned directed top-`K` graph with head/tail roles | two-stage knowledge attention and dual interaction | mean or coordinatewise max | relational node field then first moment or extrema | top-`K` instability, learned shortcut |
| HACT | cell graph, tissue graph, hard assignment | cell context, assignment transfer, tissue context | tissue sum or depth summaries | coarse hierarchical sum | assignment nullspace, segmentation bias |

## 3. The Four Axes Are Not Interchangeable

### Context `C`

Context determines which local configurations can alter a node state:

```math
\widetilde h_{iv}
=
\mathcal{C}_{\theta,v}
\left(
\left\{
h_{iu}^{(0)}:u\in\mathcal{N}_i(v)
\right\},
\text{edge data}
\right).
```

Changing `C` changes what the readout's input means. Patch-GCN's attention
weights are over spatially contextualized nodes; WiKG's mean is over nodes that
already contain a learned directed neighborhood message.

### Readout `R`

Readout determines the final collision:

```math
\mathcal{R}(\widetilde H_i)
=
\mathcal{R}(\widetilde H_i')
\Longrightarrow
\text{the task head cannot distinguish the two fields.}
```

Mean, max, attention, typed pooling, and hierarchy discard different
information. The same graph context can therefore support materially different
slide representations under different readouts.

### Geometry `G`

Geometry determines which relation the model treats as available evidence:

```text
coordinates:
    physical proximity

feature kNN:
    encoder similarity

learned top-k:
    task-conditioned directed compatibility

hierarchy:
    explicit fine-to-coarse membership
```

None is guaranteed to be the true tissue relation. Each is an inductive bias.

### Supervision `S`

The task loss determines which collisions are penalized. A classification label
can reward a shortcut in graph topology; a survival loss can reward a risk
ordering while leaving horizon calibration unconstrained. The graph should be
evaluated with a task-appropriate statistic, not only with a visually plausible
heatmap.

## 4. Design Rules

1. Specify the graph support before describing the graph neural network. A
   coordinate graph, feature graph, and learned graph are different input
   objects.

2. Write the context and readout as separate maps. A local graph softmax is not
   a slide-level attention readout.

3. State what the readout preserves. “Attention pooling” is incomplete without
   saying whether the statistic is a weighted first moment of raw or
   contextualized nodes.

4. Track count and missing-type masks. Sum pooling and typed pooling can encode
   preprocessing or population size as strongly as morphology.

5. Test topology interventions. Hold node features fixed and change `A`; if the
   output changes, the method is using geometry. That may be intended, but it
   must be measured.

6. Test the bottleneck. For HACT, perturb fine states in the nullspace of
   `B_i^T`; for WiKG, perturb near-tied top-k scores; for HEAT, perturb the
   pseudo-type map; for Patch-GCN, perturb coordinate support.

## 5. Dense Summary

Graph MIL is best described as:

```math
\boxed{
\text{slide prediction}
=
\mathcal{H}
\circ
\mathcal{R}
\circ
\mathcal{C}
\left(
\text{patch features};
\text{graph geometry and attributes}
\right)
}
```

The useful comparison is therefore not:

```text
Patch-GCN versus HEAT versus WiKG versus HACT
```

as a list of model names. It is:

```text
which relation is constructed;
which context statistic is propagated;
which graph statistic survives readout;
which supervision rewards the resulting collisions.
```
