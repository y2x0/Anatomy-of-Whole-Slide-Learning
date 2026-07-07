# Pooling Operators

This notebook family asks:

```text
What information survives aggregation?
```

Slide representations ask what mathematical object the slide is. Pooling asks
how that object is compressed into a finite slide statistic.

The generic map is:

```math
\mathcal{X}_i
\xrightarrow{\mathcal{C}}
\widetilde H_i
=
\{u_{ij}\}_{j=1}^{n_i}
\xrightarrow{\mathcal{R}}
z_i
\xrightarrow{\mathcal{H}}
\widehat y_i.
```

The pooling operator is:

```math
\mathcal{R}:\{\mathbb{R}^{d}\}^{n_i}\to\mathbb{R}^{D}.
```

It is the bottleneck where many instance states become one slide statistic.

## C/R/G/S Format

Every pooling operator should be placed in the same decomposition used by the
rest of the repository:

```math
\widetilde H_i
=
\mathcal{C}(H_i;G_i,S_i),
\qquad
z_i
=
\mathcal{R}(\widetilde H_i;G_i,S_i),
\qquad
\widehat y_i
=
\mathcal{H}(z_i).
```

For pooling notes:

```text
C:
    the context operator that creates the instance states being pooled

R:
    the readout or pooling operator itself

G:
    geometry, order, graph, hierarchy, or prototype geometry available to C or R

S:
    supervision signal that shapes scores, prototypes, pseudo labels, or risk
```

Each method should answer:

```text
1. What is C?
2. What is R?
3. What structure G is used or ignored?
4. What supervision S shapes the pooling?
5. What statistic survives in z?
6. What failure mode follows?
```

## First Chunk

```text
00_problem_setup.md
    pooling as a statistic, equivalence relation, and bottleneck

mean_pooling/
    mean as first moment
    mean after context
    mean failure modes

max_extreme_pooling/
    max as extreme statistic
    log-sum-exp and softmax limits
    sparse-positive MIL

attention_pooling/
    attention as learned measure
    gated attention and gradients
    DSMIL critical-instance attention
    attention failure modes

class_specific_attention/
    class-conditioned readouts
    CLAM instance clustering
    loss geometry
    class-specific failure modes

prototype_pooling/
    prototypes as quantized measures
    counts, residuals, and transport
    PANTHER GMM statistics
    prototype failure modes

unifying_view/
    C/R/G/S pooling decomposition
    paper placement matrix
```

## Larger Map

There are more useful pooling notes than the initial three families. The full
pooling notebook should eventually include:

```text
moment_pooling/
    first moments
    covariance and second moments
    learned moment maps
    kernel mean embeddings

mean_pooling/
    raw mean
    mean after context
    normalized versus unnormalized sums
    failure modes

max_extreme_pooling/
    hard max
    top-k pooling
    quantile pooling
    log-sum-exp limits
    generalized means

noisy_or_pooling/
    instance probability model
    bag probability as logical OR
    noisy-or gradients
    identifiability failures

attention_pooling/
    learned reweighted measure
    gated attention MIL
    attention temperature
    attention collapse

class_specific_attention/
    class-conditioned readouts
    CLAM-style instance constraints
    competing attention heads

prototype_pooling/
    soft histograms
    mixture statistics
    residual encodings
    prototype drift

distribution_pooling/
    CDF and quantile summaries
    optimal transport summaries
    MMD and kernel distances

set_transformer_pooling/
    pooling by multihead attention
    seed vectors
    inducing points
    learned queries as statistic extractors

additive_evidence_pooling/
    sum pooling
    count-like evidence
    risk accumulation
    survival readouts

robust_pooling/
    trimmed means
    winsorized summaries
    median and geometric median
    outlier resistance

hierarchical_pooling/
    region-to-slide pooling
    scale-weighted pooling
    multiscale readouts

unifying_view/
    pooling as statistical functional
    surviving statistic matrix
    paper placement matrix
    design and diagnostic checklist
```

The point is not to list operators. The point is to make each operator's
surviving statistic and failure mode mathematically visible.

## Core Distinction

Context operator:

```text
mixes information between instances
```

Pooling operator:

```text
decides what statistic of the mixed instances survives
```

A mean after a GNN is not the same object as a mean before a GNN. The formula may
look identical, but the states being averaged are different.

## Anchor Ideas

- Deep Sets: invariant functions through transform and sum.
- Attention MIL: learned weighted first moment.
- CLAM: class-specific attention plus instance constraints.
- Set Transformer: query-based pooling over permutation-equivariant states.
- Noisy-or MIL: bag probability from instance probabilities.
- Prototype pooling: morphology histogram or mixture statistic.
- Survival MIL: pooling determines which slide statistic becomes risk.
