# WSI Interface, C/R/G/S, And Failure Matrix

## C/R/G/S Placement

| Method | Context `C` | Readout `R` | Geometry `G` | Supervision `S` |
|---|---|---|---|---|
| DINO | ViT over each crop | class token and projection head | teacher-student categorical cross-entropy | global-to-global and global-to-local same-image views |
| iBOT | ViT over visible and mask tokens | class token plus masked patch tokens | online-tokenizer categorical prediction | same patch position in visible teacher and masked student |
| DINOv2 | ViT with image and patch heads | separate global and local projection heads | distillation, balanced assignments, nearest-neighbor spacing | same-image crops and masked positions |
| HIPT | nested ViTs at patch, region, and slide scales | class token at each scale | stagewise DINO then hierarchical aggregation | same patch or region under scale-specific views |
| UNI | ViT-L trained with DINOv2 | exported patch representation | DINOv2 composite geometry | Mass-100K tile distribution |
| Virchow | ViT-H/14 trained with DINOv2 | class token concatenated with mean patch token | DINOv2 with 131,072 prototype coordinates | hierarchical WSI-then-tile sampling |
| Phikon | ViT with iBOT | class and masked patch-token outputs | global distillation plus masked token prediction | histology crop and mask relation |

## Surviving Statistics

DINO exports a global crop statistic:

```math
h_{\mathrm{CLS}}(x).
```

iBOT constrains local conditional statistics:

```math
p
\left(
Z_i^t
\mid
X_O
\right).
```

Virchow exports a concatenated statistic:

```math
\left[
h_{\mathrm{CLS}};
\mathbb{E}_{j}
h_j^{\mathrm{patch}}
\right].
```

HIPT repeatedly compresses 256 tokens into one class token:

```math
256
\longrightarrow
1
\longrightarrow
256
\longrightarrow
1.
```

## Failure Matrix

| Failure | Mathematical source | Consequence |
|---|---|---|
| constant output | agreement objective admits `p(x)=c` | no discriminative representation |
| coordinate domination | one teacher output dimension wins globally | pseudo-class collapse |
| over-sharpening | very small teacher temperature | brittle low-entropy targets |
| center bias | batch mean estimates skewed training mixture | anti-collapse statistic follows organ or institution imbalance |
| teacher lag | large EMA memory | stale targets during rapid learning or domain change |
| crop semantic mismatch | local crop omits defining tissue context | incompatible global-to-local targets |
| masked ambiguity | high conditional entropy at hidden position | rare detail is averaged or ignored |
| frozen hierarchy bottleneck | lower-stage class token is non-injective | higher stage cannot recover discarded morphology |
| within-slide batch shortcut | many correlated tiles per sampled WSI | uniformity and centering mix slide identity with morphology |
| slide readout mismatch | patch objective and downstream task use different statistics | strong tile encoder does not imply sufficient WSI representation |

## Sparse-Positive Counterexample

Let a tile be positive because of one rare token `r`. If a local crop or mask
removes that token:

```math
r
\notin
V_s,
\qquad
r
\in
V_t,
```

the student is asked to predict a teacher distribution influenced by evidence
it cannot observe. The Bayes-optimal student uses correlations in surrounding
tissue rather than the rare token itself.

## Slide Sufficiency Boundary

Let patch encoder be `f` and slide task be `Y`. A necessary condition for some
downstream readout to be Bayes sufficient is:

```math
Y
\perp
\left\{
x_j
\right\}_{j=1}^{n}
\mid
\left\{
f(x_j)
\right\}_{j=1}^{n}.
```

Negative-free pretraining does not establish this condition. It optimizes view
and mask prediction under the pretraining distribution.

## Unified Failure Principle

```math
\text{tile sampling}
\longrightarrow
\text{view and mask kernel}
\longrightarrow
\text{teacher target}
\longrightarrow
\text{patch readout}
\longrightarrow
\text{slide aggregation}.
```

Each arrow is an information boundary. Anti-collapse mechanisms ensure that
something varies; they do not guarantee that the variation surviving every
boundary is the variation required by the slide task.
