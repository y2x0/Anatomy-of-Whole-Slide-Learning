# C/R/G/S For Multimodal Alignment

This note places the derived methods in the same decomposition used throughout
the repository.

## Definitions

For paired or bagged observations, write:

```math
\widehat y
=
\mathcal{H}
\left(
\mathcal{R}
\left(
\mathcal{C}(X,T)
\right);
\mathcal{G},
\mathcal{S}
\right).
```

The components are:

```math
\mathcal{C}
=
\text{within-modality and cross-modal context operators},
```

```math
\mathcal{R}
=
\text{readout that determines what information survives},
```

```math
\mathcal{G}
=
\text{geometry in which modalities are compared},
```

```math
\mathcal{S}
=
\text{observed or generated supervision relation}.
```

## Method Placement

| Method | Context `C` | Readout `R` | Geometry `G` | Supervision `S` |
|---|---|---|---|---|
| CLIP | separate image and text towers | one normalized global vector per object | spherical bilinear compatibility | diagonal web pair identity |
| PLIP | CLIP towers specialized to pathology | one normalized global vector | spherical compatibility | OpenPath tweets, replies, and retrieval-selected pairs |
| CONCH | image ViT, text transformer, cross-modal decoder | one contrast token plus 256 caption tokens | contrastive sphere plus autoregressive token likelihood | cleaned figure-caption pairs |
| CPLIP | PLIP encoders over constructed bags | log-sum-exp over Cartesian-product instance scores | hierarchical bag and instance softmax geometry | dictionary, GPT-3, and retrieval-generated bag relation |
| PathCLIP benchmark | fixed evaluated encoders | zero-shot prototypes or retrieval ranks | corruption-induced angular drift | no new training relation |

## Surviving Statistics

CLIP and PLIP retain one global direction:

```math
x
\longmapsto
u(x)
\in
\mathbb{S}^{d-1}.
```

CONCH retains two image statistics for two losses:

```math
H
\longmapsto
\left(
h^{\mathrm{ctr}},
Z^{\mathrm{cap}}
\right).
```

CPLIP retains a soft maximum over pair similarities:

```math
S_{ij}
=
\log
\sum_{m,n}
\exp
\left(
u_{im}^{\top}v_{jn}/\sigma
\right).
```

CONCH WSI inference retains a class-specific upper-tail mean:

```math
S_{ic}^{(K)}
=
\frac{1}{K}
\sum_{r=1}^{K}
s_{i(r)c}.
```

These are mathematically different objects even when every method is described
informally as image-text alignment.

## Failure Matrix

| Failure | Formal source | Most exposed methods | What breaks |
|---|---|---|---|
| duplicate-caption contradiction | one-positive row softmax | CLIP, PLIP | interchangeable positives repel or impose irreducible loss |
| corpus selection bias | conditioning on inclusion event `A=1` | PLIP, CONCH | learned density ratio targets selected population |
| imported encoder geometry | pretrained model constructs pairs | PLIP PathLAION, CONCH cleaning, CPLIP | support and pair relation inherit teacher blind spots |
| visually ungrounded language | case or social text predicts pair identity | PLIP, CONCH | compatibility need not imply localized entailment |
| cardinality bias | unnormalized bag partition `A_ij` | CPLIP | larger bags gain logit mass |
| positive-bag contamination | invalid pairs receive exponential mass | CPLIP | one high-scoring bad pair can dominate |
| prompt boundary rotation | text prototype changes | all zero-shot uses | class boundary and selected tiles change |
| upper-tail blindness | top-k pooling | CONCH WSI evaluation | prevalence and non-top-k structure disappear |
| corruption drift | image encoder Jacobian and information loss | PathCLIP benchmark | classification margin or retrieval rank changes |

## Design Axes That Are Not Equivalent

Changing pair data modifies `S`:

```math
p_{\mathrm{obs}}(x,t)
\longrightarrow
p'_{\mathrm{obs}}(x,t).
```

Changing from one vector to local tokens modifies `R`:

```math
h
\longrightarrow
Z_{1:L}.
```

Adding caption cross-attention modifies `C`:

```math
w_{<r}
\longrightarrow
\mathrm{CrossAttn}
\left(
w_{<r},Z_{1:L}
\right).
```

Changing cosine similarity to bag log-sum-exp modifies `G` and `R` jointly:

```math
u^{\top}v
\longrightarrow
\log
\sum_{m,n}
\exp
\left(
u_m^{\top}v_n/\sigma
\right).
```

Calling all four changes better alignment hides which inductive bias actually
changed.

## Unified Failure Principle

The learned representation is constrained by the composition:

```math
\text{pair construction}
\longrightarrow
\text{context}
\longrightarrow
\text{readout}
\longrightarrow
\text{comparison geometry}.
```

No downstream objective can recover information removed before it, and no
amount of architectural capacity can make an incorrect supervision relation
become correct merely by fitting it more accurately.
