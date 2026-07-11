# C/R/G/S Cross-Slide Design Matrix

## Method Placement

| Method | Context `C` | Readout `R` | Geometry `G` | Supervision `S` |
|---|---|---|---|---|
| SCL-WC CDA | class-specific patch scoring | weighted slide feature plus top/bottom-k supports | MIL and binary classification | slide label generates pseudo-instance targets |
| SCL-WC PNM | class-agnostic gate | complementary weighted first moments | positive/negative classification | assumed partition of abnormal and normal tissue |
| SCL-WC WSCL | cross-slide memory lookup | top-k, bottom-k, and hard-negative subbags | multi-positive temperature softmax | same disease class after attention selection |
| SC-MIL | attention MIL within each slide | one bag embedding then projection | bag-level supervised contrast plus CE | same observed slide class |
| CAMCSA | CAM-guided cross-slide mixing | selected significant instances | mixed-sample classifier geometry | interpolated bag targets |

## Comparison Scale

Patch-selected cross-slide contrast:

```math
z_{ij}
\longleftrightarrow
z_{ab}.
```

Bag-level contrast:

```math
b_i
\longleftrightarrow
b_j.
```

Mixed-bag augmentation:

```math
\left(
\mathcal{X}_i,Y_i
\right),
\left(
\mathcal{X}_j,Y_j
\right)
\longrightarrow
\left(
\widetilde{\mathcal{X}}_{ij},
\widetilde Y_{ij}
\right).
```

These operators cannot be compared merely by saying they use information from
multiple slides.

## Surviving Statistics

SCL-WC survives through selected order statistics and memory-bank geometry:

```math
\left(
\mathrm{TopK}(A),
\mathrm{BottomK}(A)
\right).
```

SC-MIL survives through one attention-pooled vector:

```math
b_i
=
\sum_j
\alpha_{ij}h_{ij}.
```

PNM survives through two complementary first moments. CAMCSA survives through
a synthetic selected-instance mixture.

## Design Matrix

| Design question | Choice | Mathematical consequence |
|---|---|---|
| What is a positive? | same label | compresses all within-class heterogeneity |
| What is compared? | bag vector | inherits readout collisions |
| What is compared? | selected patches | inherits attention support error |
| How are rare classes supplied? | memory bank | adds staleness and historical bias |
| How are hard negatives found? | high attention on negative slides | may mine shortcuts or false negatives |
| How are slides combined? | feature mixing | creates synthetic distribution, not contrastive geometry |
| What is the split unit? | slide rather than patient | permits patient leakage |

## Complexity

For batch size `B` and bag embedding dimension `d`, SC-MIL contrast costs:

```math
\Theta
\left(
B^2d
\right).
```

For `k` selected patches in `M` memory bags, SCL-WC patch comparisons cost:

```math
\Theta
\left(
k^2Md
\right)
```

per anchor-bag comparison up to implementation batching. Selection reduces the
full WSI combinatorics but makes top-k support a hard information bottleneck.

## Unified Failure Principle

```math
\text{weak slide label}
\longrightarrow
\text{readout or selected support}
\longrightarrow
\text{cross-slide relation}
\longrightarrow
\text{embedding geometry}.
```

Cross-slide contrast can improve inter-slide organization only for information
that survives the readout or enters selected support. It cannot make an
unidentified patch label become observed truth.
