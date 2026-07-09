# Heterogeneous Graph Failure Modes

Chan et al. make WSI graph learning more expressive by adding:

```text
patch node pseudo-types
continuous edge attributes
HEAT message passing
PL pooling
node-removal localization
```

Each design choice creates a precise failure mode.

## 1. Patch Type Is A Majority Vote

The node type is:

```math
\tau_i(v)
=
\mathrm{mode}
\left(
\{\text{nucleus types in patch }v\}
\right).
```

If a patch contains mixed nuclei:

```text
neoplastic minority
inflammatory majority
```

then:

```math
\tau_i(v)
=
\mathrm{inflammatory}
```

and the minority neoplastic signal does not survive as a node type.

The node feature may still contain visual evidence, but the heterogeneity
channel `tau(v)` has collapsed the within-patch distribution to one category.

## 2. Pseudo-Label Noise Is Structural Noise

The node type comes from pretrained HoVer-Net, not from ground-truth patch
labels.

If HoVer-Net misclassifies nuclei, then:

```math
\tau_i(v)
```

is wrong.

This affects:

```text
type-specific HEAT projections
PL pooling cluster membership
type-wise graph-level summaries
```

So pseudo-label error is not just feature noise. It changes which operators act
on the node.

## 3. Feature-Space kNN Is Not Spatial Adjacency

The graph support is:

```math
E_i
=
\{(u,v):u\in\mathcal{K}_k(v)\}
```

under the source-neighbor-to-target orientation used in these notes.

Two patches may be far apart in the WSI but adjacent in the graph:

```math
(u,v)\in E_i
\quad
\text{and}
\quad
\|c_u-c_v\|_2
\gg
0.
```

This can be useful for encoder-space relations. But it means the model is not
purely modeling physical tissue adjacency.

If the task depends on local spatial arrangement, feature-kNN can connect the
wrong support.

## Spatial Layout Counterexample

Consider two slides whose typed feature graphs are isomorphic. That is, there
exists a bijection:

```math
\pi:
V_i
\to
V_j
```

such that node features match:

```math
h_{iv}
=
h_{j,\pi(v)},
```

pseudo-labels match:

```math
\tau_i(v)
=
\tau_j(\pi(v)),
```

feature-kNN support matches:

```math
(u,v)\in E_i
\Longleftrightarrow
(\pi(u),\pi(v))\in E_j,
```

and edge attributes match:

```math
\phi_i(u,v)
=
\phi_j(\pi(u),\pi(v)).
```

But suppose the coordinates are not equivalent under that same bijection:

```math
c_{iv}
\neq
c_{j,\pi(v)}
```

Any HEAT model that only receives:

```math
\left(
H,
E^{\mathrm{feature}},
\tau,
\phi
\right)
```

must assign isomorphic node representations and the same graph representation
after permutation-invariant readout. Therefore, pure feature-kNN heterogeneity
cannot distinguish spatial rearrangements unless coordinates are encoded in
node features, edge attributes, or the graph support.

## 4. Pearson Edge Attributes Are Encoder-Dependent

The edge attribute is:

```math
r_{ist}
=
\rho_{\mathrm{Pearson}}
\left(
h_{is}^{(0)},
h_{it}^{(0)}
\right).
```

Therefore edge attributes depend on the feature encoder.

If KimiaNet embeddings shift under:

```text
scanner variation
stain variation
tissue processing artifacts
domain shift
```

then both:

```text
feature-kNN edges
Pearson edge attributes
```

can change even if the underlying tissue relationship is similar.

## 5. Continuous Edge Attributes Do Not Equal Relation Types

HGT and R-GCN often use discrete relation types:

```math
r(e)
\in
\{1,\dots,R\}.
```

Chan et al. use continuous Pearson-like edge attributes:

```math
\phi(e)
\in
\mathbb{R}.
```

This is more flexible, but less semantically stable. A scalar correlation does
not say:

```text
tumor-to-stroma
immune-to-tumor
dead-to-neoplastic
```

It only gives a learned feature-similarity signal.

## 6. PL Pooling Inherits Teacher Bias

PL pooling assumes:

```math
V_{ia}
=
\{v:\tau_i(v)=a\}
```

has consistent meaning across WSIs.

That consistency comes from HoVer-Net and PanNuke pretraining. If the target
cohort has nuclei appearances not well represented by the teacher model, then
the fixed pseudo-label clusters become biased.

The paper reports that HoverNet pseudo-labels outperform unsupervised K-means
types in a COAD classification experiment, but this does not remove the general
teacher-domain-shift risk.

## 7. Empty Or Rare Types Are Underspecified

For a type:

```math
a\in\mathcal{T},
```

a slide may have:

```math
V_{ia}
=
\varnothing.
```

Then:

```math
z_{ia}
=
\mathcal{R}_a(\varnothing)
```

must be defined by implementation convention.

Rare node types also create high-variance type summaries:

```math
|V_{ia}|
\text{ small}
\quad
\Longrightarrow
\quad
z_{ia}
\text{ noisy}.
```

## 8. Localization Is A Model Perturbation

The localization score compares:

```math
M_\theta(G_i)
\quad
\text{and}
\quad
M_\theta(G_i\setminus\{v\}).
```

Removing one node can alter:

```text
neighbor messages
type-wise pooling
graph-level class evidence
```

Under the fixed-graph deletion convention used in these notes, node removal
deletes incident edges but does not rebuild feature-kNN support. Rebuilding the
support would define a different intervention.

So the score is a useful graph perturbation diagnostic, but it is not direct
biological causality.

## Concrete Stress Tests

```text
node-type noise:
    randomly flip tau(v) for a percentage of nodes and measure performance drop

majority-vote stress:
    replace majority type with a type histogram per patch and compare

feature-kNN versus coordinate-kNN:
    swap graph support to spatial kNN and test which tasks change

Pearson ablation:
    remove phi(e), binarize it, or replace it with distance and compare

empty-type masking:
    test zero vector, learned missing-type vector, and masked graph readout

teacher shift:
    compare HoVer-Net types, K-means types, and manually audited type subsets

causal score sign:
    verify whether visualization uses Delta or negative Delta as importance
```

## C/R/G/S Diagnostic Questions

```text
G:
    Are node pseudo-types reliable?
    Does feature-kNN represent the relation the task needs?

C:
    Does edge-modulated attention use phi(e), or can it ignore it?
    Are type-specific projections overfitting rare types?

R:
    Are pseudo-label clusters semantically stable across slides?
    What happens when a type is absent?

S:
    Does slide-level supervision actually constrain the node-type and
    edge-attribute assumptions, or only the final graph statistic?
```
