# C/R/G/S, Surviving Geometry, And Failure Matrix

Contrastive objectives differ in the supervision object that defines
similarity, the mechanism preventing collapse, and the geometry directly
constrained by training.

Primary anchors:

- CPC: https://arxiv.org/abs/1807.03748
- SimCLR: https://arxiv.org/abs/2002.05709
- MoCo: https://arxiv.org/abs/1911.05722
- Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
- BYOL: https://arxiv.org/abs/2006.07733
- SimSiam: https://arxiv.org/abs/2011.10566
- Barlow Twins: https://arxiv.org/abs/2103.03230
- VICReg: https://arxiv.org/abs/2105.04906

## Common Representation Pipeline

Let two related observations be encoded as:

```math
h^{(b)}
=
\mathcal{C}_{\theta}
\left(
V^{(b)}
\right),
\qquad
b\in\{1,2\}.
```

The compared object is:

```math
z^{(b)}
=
\mathcal{R}_{\phi}
\left(
h^{(b)}
\right).
```

The objective has the generic form:

```math
\mathcal{L}
=
\mathcal{S}
\left(
z^{(1)},
z^{(2)},
\mathcal{K},
\mathcal{T}
\right),
```

where `K` denotes candidate or batch context and `T` denotes a generated target
such as a positive index, stopped teacher vector, or moment constraint.

## Positive Relation As A Graph

For a finite dataset of views, define a positive-relation graph:

```math
G_{+}
=
(V,E_{+}),
```

where:

```math
(i,j)
\in
E_{+}
```

means the training objective directly attracts objects `i` and `j`.

The transitive closure induces connected components:

```math
\mathcal{C}_1,
\ldots,
\mathcal{C}_M.
```

Pure attraction admits mappings constant on each component:

```math
z_i
=
c_m,
\qquad
i\in\mathcal{C}_m.
```

If the positive graph is connected, pure attraction admits global collapse.
Explicit negatives, stop-gradient dynamics, or moment constraints determine
whether and how components remain distinguishable.

## InfoNCE C/R/G/S

```text
\mathcal{G}:
    one positive conditional and N-1 proposal candidates

\mathcal{C}:
    anchor and key encoders

\mathcal{R}:
    projected embedding and optional normalization

\mathcal{S}:
    N-way positive-index cross-entropy
```

The surviving geometry is relative density-ratio geometry:

```math
f^{\star}(a,k)
=
\log
\frac{p(k\mid a)}{\nu(k)}
+
c(a).
```

## SimCLR C/R/G/S

```text
\mathcal{G}:
    two-view augmentation graph and in-batch candidate support

\mathcal{C}:
    shared base encoder

\mathcal{R}:
    nonlinear projector followed by unit normalization

\mathcal{S}:
    symmetric NT-Xent positive-index loss
```

The loss directly constrains angles in projection space:

```math
u_i^{\top}u_j.
```

The retained encoder representation `h_i` survives downstream, not necessarily
the normalized projected vector `u_i`.

## MoCo C/R/G/S

```text
\mathcal{G}:
    current positive key plus a time-mixed queue dictionary

\mathcal{C}:
    gradient-updated query encoder and EMA key encoder

\mathcal{R}:
    normalized query and key embeddings

\mathcal{S}:
    dictionary lookup cross-entropy
```

The surviving geometry depends on queue age and key-encoder consistency in
addition to the InfoNCE critic.

## Supervised Contrastive C/R/G/S

```text
\mathcal{G}:
    class-induced positive sets in a multiview batch

\mathcal{C}:
    shared encoder

\mathcal{R}:
    normalized projection vectors

\mathcal{S}:
    equal-weight outside-log positive average
```

The attractive statistic is:

```math
\overline z_{P(i)}
=
\frac{1}{|P(i)|}
\sum_{p\in P(i)}z_p.
```

Fine-grained variation within each declared class is pressured toward a shared
class-conditioned geometry.

## BYOL And SimSiam C/R/G/S

```text
\mathcal{G}:
    paired views and directional online-target relation

\mathcal{C}:
    online/EMA encoders for BYOL or shared encoder for SimSiam

\mathcal{R}:
    projector, asymmetric predictor, and normalization

\mathcal{S}:
    stopped target vector
```

The surviving relation is predictive alignment with a target representation.
No explicit pairwise repulsion identifies inter-example geometry.

## Barlow Twins And VICReg C/R/G/S

```text
\mathcal{G}:
    paired rows and batch-coordinate relation

\mathcal{C}:
    twin or shared-weight encoders

\mathcal{R}:
    batch embedding matrices and their moments

\mathcal{S}:
    cross-correlation identity or invariance/variance/covariance targets
```

What survives is a batch-moment geometry:

```math
\text{paired alignment}
+
\text{coordinate spread}
+
\text{reduced redundancy}.
```

## Objective Matrix

| Objective | Positive mechanism | Explicit negatives | Anti-collapse mechanism | Directly constrained object |
|---|---|---:|---|---|
| InfoNCE/CPC | conditional future or paired key | yes | candidate classification | critic density ratio |
| SimCLR | two augmentations | yes | in-batch repulsion | normalized projector angles |
| MoCo | paired query/key views | yes | queued repulsion | query versus historical-key geometry |
| SupCon | same observed class | yes | cross-class candidate repulsion | class-positive projection geometry |
| BYOL | online view predicts EMA target view | no | predictor, stop-gradient, EMA dynamics | normalized prediction-target alignment |
| SimSiam | one view predicts stopped other view | no | predictor and stop-gradient dynamics | normalized prediction-target alignment |
| Barlow Twins | paired views | no | identity cross-correlation target | cross-view coordinate correlations |
| VICReg | paired views | no | explicit variance floor and covariance penalty | paired distances and per-branch moments |

## Surviving Geometry Versus Surviving Representation

The objective geometry lives at:

```math
z
=
\mathcal{R}_{\phi}(h).
```

The downstream system may retain:

```math
h
=
\mathcal{C}_{\theta}(v).
```

Therefore every method should report both:

```text
pretraining comparison space
downstream retained space
```

Good contrastive loss values do not prove that the retained representation has
the same pairwise distances, calibration, or neighborhood structure.

## Failure Matrix

| Failure | Mathematical source | Diagnostic |
|---|---|---|
| false positive | positive graph joins semantically different objects | perturb pairing rule and inspect retained factors |
| false negative | high-similarity semantic match enters denominator | estimate class/morphology collision rate |
| proposal bias | negatives follow `nu` rather than population marginal | report sampler and density-ratio target |
| MI saturation | lower bound cannot exceed `log N` | compare candidate count with expected information scale |
| temperature collapse | Gibbs mass concentrates on very few candidates | entropy and maximum candidate probability |
| queue staleness | keys come from historical encoders | accuracy versus key age and momentum |
| representation collapse | attraction has constant solution | coordinate variance and effective rank |
| dimensional collapse | coordinates become redundant | covariance spectrum and off-diagonal energy |
| projector mismatch | pretraining geometry differs from retained geometry | compare neighborhoods before and after projector |
| batch-moment noise | moments estimated from small correlated batches | bootstrap variance across batch compositions |

## Pathology Design Boundary

The objective cannot decide which pathology relation should define similarity.
That decision enters through:

```text
augmentation
same patch or same slide
same patient
same diagnosis
same morphology cluster
image-text pairing
graph augmentation
```

Each choice defines a different positive graph and therefore a different
invariance class.

## Dense Summary

Contrastive learning is a geometry-design problem:

```math
\boxed{
\text{sampling relation}
\to
\text{objective}
\to
\text{projection geometry}
\to
\text{retained representation}
}
```

Failure at any arrow can produce a mathematically optimized objective with the
wrong biological similarity.
