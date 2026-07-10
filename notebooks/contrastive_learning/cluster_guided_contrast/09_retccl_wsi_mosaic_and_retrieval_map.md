# RetCCL WSI Mosaic And Retrieval Map

RetCCL is not only a patch encoder. Its second stage represents a slide by a
selected mosaic of patch features, performs patch-level memory lookup, and
aggregates retrieval bags into slide results.

Primary anchors:

- RetCCL paper: https://pubmed.ncbi.nlm.nih.gov/36270093/
- Official feature-extraction repository:
  https://github.com/Xiyue-Wang/RetCCL

## Source Boundary

The paper specifies the full database-construction and WSI-query algorithms.
The official repository releases the pretrained CCL feature extractor and
feature-extraction utilities, then points readers to a third-party
reproduction for WSI retrieval. The equations below reconstruct the paper
algorithms; they do not claim that the complete retrieval pipeline is present
in the official repository.

## Two Stages

The complete method is:

```math
\text{unlabeled pathology patches}
\xrightarrow{
\mathcal{L}_{\mathrm{CCL}}
}
f_{\widehat\theta}
```

followed by:

```math
\text{WSI archive}
\xrightarrow{
f_{\widehat\theta}
}
\text{mosaic database}
\xrightarrow{
\text{query, rank, aggregate}
}
\text{retrieved WSIs}.
```

The contrastive objective trains the patch feature extractor. The retrieval
stage is a separate, nonparametric slide operator.

## Paper-Level Database Parameters

Algorithm 1 sets:

```text
D_s = 16:
    downsampling factor for foreground segmentation

MPP = 1.0:
    resolution used for patch extraction

S_p = 512:
    patch side length

K_1 = 9:
    number of feature clusters

R = 0.2:
    second spatial-clustering ratio
```

These values define the reported mosaic construction; they are not implied by
the contrastive loss.

## Foreground And Patch Map

For slide image `I_i`, foreground segmentation gives:

```math
\Omega_i
=
\mathrm{Segment}
\left(
I_i,D_s
\right).
```

Patching at the stated resolution and size gives:

```math
\mathcal{P}_i
=
\left\{
p_{ij}
\right\}_{j=1}^{n_i}
=
\mathrm{Patch}
\left(
\Omega_i,
\mathrm{MPP},
S_p
\right).
```

Each patch has a coordinate:

```math
r_{ij}
\in
\mathbb{R}^{2}.
```

The pretrained CCL encoder produces:

```math
z_{ij}
=
f_{\widehat\theta}
\left(
p_{ij}
\right)
\in
\mathbb{R}^{d}.
```

## First Clustering: Feature Diversity

Feature k-means partitions:

```math
\left\{
z_{ij}
\right\}_{j=1}^{n_i}
```

into `K_1` groups:

```math
\mathcal{F}_{i,1},
\ldots,
\mathcal{F}_{i,K_1}.
```

This stage seeks diversity in learned appearance space.

## Second Clustering: Spatial Coverage

Within each feature cluster, the corresponding coordinates are clustered:

```math
\left\{
r_{ij}
:
z_{ij}
\in
\mathcal{F}_{i,k}
\right\}
\xrightarrow{
\mathrm{SpatialKMeans}(\cdot,R)
}
\mathcal{M}_{i,k}.
```

The returned representative patches or features are united:

```math
\mathcal{M}_i
=
\bigcup_{k=1}^{K_1}
\mathcal{M}_{i,k}.
```

RetCCL calls this selected set a mosaic.

## The Slide Object Is A Selected Set

The database representation of slide `i` is:

```math
\mathcal{M}_i
=
\left\{
\left(
z_{ir},
\mathrm{meta}_{ir}
\right)
\right\}_{r=1}^{m_i},
```

where metadata associates a representative patch with its source slide and
diagnosis.

It is not one vector:

```math
\mathcal{M}_i
\notin
\mathbb{R}^{d}.
```

It is a sparse queryable memory object:

```math
\mathcal{M}_i
\subset
\mathbb{R}^{d}
\times
\mathcal{Y}
\times
\mathcal{I}.
```

Feature and spatial clustering decide which patch evidence survives.

## Archive Database

Across archive slides:

```math
\mathcal{G}
=
\bigcup_{i=1}^{N_{\mathrm{WSI}}}
\mathcal{M}_i.
```

The paper's Algorithm 1 returns `\mathcal{G}` as the mosaic database for
retrieval.

## Query Slide And Retrieval Bags

Let a query slide yield `k` query patches:

```math
\mathcal{P}^{q}
=
\left\{
P_1,
\ldots,
P_k
\right\}.
```

Each query patch retrieves `t` database patches:

```math
\mathbb{B}_i
=
\left\{
b_i^1,
\ldots,
b_i^t
\right\}.
```

The query stage therefore produces a bag of bags:

```math
\mathrm{Bag}
=
\left\{
\mathbb{B}_1,
\ldots,
\mathbb{B}_k
\right\}.
```

This hierarchy is:

```math
\text{query WSI}
\longrightarrow
\text{query patches}
\longrightarrow
\text{retrieved patch bags}
\longrightarrow
\text{source WSI votes}.
```

## Cosine Similarity

For query feature `z(P_i)` and retrieved feature `z(b_i^j)`:

```math
d_i^j
=
\frac{
z(P_i)^{\top}z(b_i^j)
}{
\lVert z(P_i)\rVert_2
\lVert z(b_i^j)\rVert_2
}
\in
\left[
-1,1
\right].
```

RetCCL maps this to a nonnegative score:

```math
s_i^j
=
\frac{
d_i^j+1
}{
2
}
\in
\left[
0,1
\right].
```

## Diagnosis-Weighted Bag Probability

Let:

```math
y_i^j
```

be the database diagnosis attached to retrieved patch `b_i^j`, and let:

```math
w_y
```

be the paper's database weight for diagnosis `y`.

For diagnosis type `m`, RetCCL computes:

```math
p_{i,m}
=
\frac{
\sum_{j=1}^{t}
\delta
\left(
y_i^j,m
\right)
w_{y_i^j}
\left(
d_i^j+1
\right)/2
}{
\sum_{j=1}^{t}
w_{y_i^j}
\left(
d_i^j+1
\right)/2
}.
```

When the denominator is positive:

```math
p_{i,m}
\ge
0,
```

and over diagnoses present in the bag:

```math
\sum_m
p_{i,m}
=
1.
```

This is a similarity-weighted categorical distribution over database
diagnoses. The database weights alter the prior contribution of each diagnosis.

If every retrieved cosine similarity equals `-1`, the denominator is
zero. A practical implementation needs a numerical fallback for this degenerate
case.

## Retrieval-Bag Entropy

If bag `i` contains `u_i` diagnosis types:

```math
\mathrm{Ent}_i
=
-
\sum_{m=1}^{u_i}
p_{i,m}
\log
p_{i,m}.
```

Its bounds are:

```math
0
\le
\mathrm{Ent}_i
\le
\log
u_i.
```

Low entropy means the retrieved patch evidence concentrates on a small number
of diagnoses. High entropy means diagnostic disagreement within the bag.

RetCCL reorders retrieval bags using this uncertainty statistic.

## Supervision Enters At Ranking Time

The CCL encoder is self-supervised, but:

```math
p_{i,m}
```

uses archive diagnosis labels:

```math
y_i^j.
```

Therefore the full retrieval pipeline is not label-free. It contains:

```math
\text{self-supervised feature learning}
+
\text{diagnosis-aware nonparametric ranking}.
```

This distinction matters when attributing performance or interpretability.

## Low-Quality Bag Filter

Let:

```math
\mathrm{AveTop}
\left(
\mathbb{B}_i
\right)
```

be the mean of the top five cosine similarities in bag `i`. Algorithm 2
defines:

```math
\eta
=
\frac{1}{k}
\sum_{i=1}^{k}
\mathrm{AveTop}
\left(
\mathbb{B}_i
\right).
```

Bags with low top-five similarity relative to this criterion are removed.
Writing the retained index set:

```math
\mathcal{I}_{\mathrm{keep}}
=
\left\{
i
:
\mathrm{AveTop}
\left(
\mathbb{B}_i
\right)
\ge
\eta
\right\}
```

makes the threshold operation explicit.

The threshold depends on the query's own bag-quality distribution. It is not a
fixed archive-wide calibration.

## Top-Five Patch Readout

For each retained bag, keep:

```math
\widetilde{\mathbb{B}}_i
=
\left\{
b_i^{(1)},
\ldots,
b_i^{(5)}
\right\}
```

ordered by retrieval quality after ranking.

Let `\mathrm{src}(b)` be the source WSI of database patch `b`. A
bag-level WSI vote can be written:

```math
W_i
=
\arg\max_{r}
\sum_{b\in\widetilde{\mathbb{B}}_i}
\mathbf{1}
\left[
\mathrm{src}(b)=r
\right].
```

The slide result aggregates these patch-bag votes:

```math
\mathrm{WSIRet}
=
\left\{
W_i
:
i\in\mathcal{I}_{\mathrm{keep}}
\right\},
```

then returns the top-ranked WSIs.

## What Information Survives

Mosaic construction retains:

```math
\text{representatives across feature clusters}
+
\text{representatives across spatial subclusters}.
```

Query aggregation retains:

```math
\text{top local matches}
+
\text{diagnosis concentration}
+
\text{source-slide vote count}.
```

It does not reduce the slide to a first moment:

```math
\frac{1}{n_i}
\sum_j z_{ij}.
```

The surviving slide statistic is a selected retrieval relation between query
patches and an external archive.

## Geometry Limitation

Spatial coordinates are used to select a diverse mosaic, but the final
retrieval comparison uses patch-feature cosine similarity. The pipeline does
not construct a graph preserving pairwise spatial adjacency among selected
patches.

Two slides can share similar selected feature sets while arranging those
features differently:

```math
\left\{
z_{aj}
\right\}
\approx
\left\{
z_{bj}
\right\},
```

yet:

```math
\left\{
\lVert r_{aj}-r_{ak}\rVert_2
\right\}_{j,k}
\ne
\left\{
\lVert r_{bj}-r_{bk}\rVert_2
\right\}_{j,k}.
```

Spatial sampling improves coverage without making layout itself the retrieval
metric.

## Complexity

Let `N_P` be the number of archive patches before mosaic selection and
`M` the total number of retained mosaic patches.

Offline feature extraction costs:

```math
\Theta
\left(
N_P
\cdot
\mathrm{encoder\ cost}
\right).
```

Naive exact query search for `k` query patches costs:

```math
\Theta
\left(
kMd
\right).
```

Retaining mosaics reduces memory and search relative to all archive patches,
but performance depends on whether feature-spatial selection preserves rare
diagnostic regions.

## C/R/G/S Placement

```text
\mathcal{G}:
    archive patch memory, within-slide feature clusters, within-cluster spatial
    clusters, and query-to-database nearest-neighbor edges

\mathcal{C}:
    CCL patch encoder, dual mosaic construction, cosine search, diagnosis-aware
    entropy ranking, and quality filtering

\mathcal{R}:
    selected mosaic; then top-five patch bags and source-slide votes

\mathcal{S}:
    self-supervised CCL during feature learning, followed by archive diagnosis
    metadata during ranking
```

## Failure Principle

RetCCL succeeds only if two selections preserve the right evidence:

```math
\text{slide}
\longrightarrow
\text{mosaic}
```

must retain the diagnostic patch, and:

```math
\text{retrieval bag}
\longrightarrow
\text{top five votes}
```

must retain its correct archive matches. Either bottleneck can erase a rare
signal even with a strong patch encoder.
