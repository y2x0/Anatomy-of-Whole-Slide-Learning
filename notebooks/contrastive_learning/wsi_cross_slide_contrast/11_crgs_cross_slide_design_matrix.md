# C/R/G/S Cross-Slide Design Matrix

## Method Placement

| Method | Context `C` | Readout `R` | Geometry `G` | Supervision `S` |
|---|---|---|---|---|
| SCL-WC CDA | class-specific patch scoring | weighted slide feature plus top/bottom-k supports | MIL and binary classification | slide label generates pseudo-instance targets |
| SCL-WC PNM | class-agnostic gate | complementary weighted first moments | positive/negative classification | assumed abnormal/normal partition |
| SCL-WC WSCL | cross-slide memory lookup | selected positive, negative, and hard-negative subbags | multi-positive temperature softmax | same disease class after attention selection |
| SC-MIL | attention MIL within each slide | one bag embedding then projection | bag-level supervised contrast plus CE | same observed slide class |
| CAMCSA | CAM-guided cross-slide mixing | selected significant instances | mixed-sample classifier geometry | interpolated bag targets |

## Comparison Scale

```math
z_{ij}
\longleftrightarrow
z_{ab}
```

for patch-selected contrast,

```math
b_i
\longleftrightarrow
b_j
```

for bag contrast, and:

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
\right)
```

for mixed-bag augmentation.

## Surviving Statistics

SCL-WC retains selected order statistics:

```math
\left(
\mathrm{TopK}(A),
\mathrm{BottomK}(A)
\right).
```

SC-MIL retains one attention-pooled vector:

```math
b_i
=
\sum_j
\alpha_{ij}h_{ij}.
```

PNM retains two complementary first moments. CAMCSA retains a synthetic
selected-instance mixture.

## Complexity

SC-MIL bag contrast costs:

```math
\Theta
\left(
B^2d
\right).
```

For `k` selected patches and `M` memory bags, SCL-WC comparison costs on the
order of:

```math
\Theta
\left(
k^2Md
\right)
```

per anchor-bag collection. Selection reduces WSI combinatorics while creating
a hard information bottleneck.

## Failure Principle

```math
\text{weak slide label}
\longrightarrow
\text{readout or selected support}
\longrightarrow
\text{cross-slide relation}
\longrightarrow
\text{embedding geometry}.
```

Cross-slide organization can use only information surviving the readout or
entering selected support.
