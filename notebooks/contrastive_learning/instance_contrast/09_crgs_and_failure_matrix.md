# Instance Contrast C/R/G/S And Failure Matrix

DSMIL and CTransPath both use pathology-specific contrastive pretraining, but
their positive graphs, context operators, and surviving geometries are
different.

Primary anchors:

- DSMIL: https://arxiv.org/abs/2011.08939
- CTransPath: https://pubmed.ncbi.nlm.nih.gov/35952419/
- SimCLR: https://arxiv.org/abs/2002.05709

## Common Composition

For patch `x`:

```math
x
\xrightarrow{
\mathcal{A}
}
\mathcal{V}(x)
\xrightarrow{
\mathcal{C}_{\theta}
}
h
\xrightarrow{
\mathcal{R}_{\phi}
}
z
\xrightarrow{
\mathcal{S}
}
\mathcal{L}_{\mathrm{patch}}.
```

For slide `i`:

```math
\left\{
x_{ij}
\right\}_{j=1}^{n_i}
\xrightarrow{
f_{\widehat\theta}
}
\left\{
h_{ij}
\right\}_{j=1}^{n_i}
\xrightarrow{
\mathcal{M}_{\psi}
}
\widehat y_i.
```

The first composition learns patch similarity. The second learns slide
prediction.

## DSMIL Pretraining C/R/G/S

```text
\mathcal{G}:
    two stochastic views per patch identity; all other minibatch views are
    denominator candidates

\mathcal{C}:
    ResNet-18 patch encoder, trained separately at each magnification

\mathcal{R}:
    SimCLR projection and unit normalization during pretraining; retained
    backbone feature for MIL

\mathcal{S}:
    symmetric same-patch NT-Xent
```

Its central statistic is:

```math
u_{n,1}^{\top}u_{n,2}
```

relative to the in-batch softmax denominator.

## CTransPath Pretraining C/R/G/S

```text
\mathcal{G}:
    paired augmented views, original-view memory query, top-S target-memory
    pseudo-positive edges, and minibatch negative candidates

\mathcal{C}:
    online CNN-Swin CTransPath plus EMA target and shared-target paths

\mathcal{R}:
    local-global patch feature followed by a linear projection

\mathcal{S}:
    symmetric negative log total probability assigned to S+1 positives
```

Its fixed-support attractive statistic is:

```math
\overline p_{\alpha}
=
\sum_{p\in\mathcal{P}}
\frac{
\exp
\left(
z^{\top}p/\tau
\right)
}{
\sum\limits_{r\in\mathcal{P}}
\exp
\left(
z^{\top}r/\tau
\right)
}
p.
```

## Downstream C/R/G/S

### DSMIL

```text
\mathcal{G}:
    WSI bag and deterministic multiscale parent relation

\mathcal{C}:
    critical-instance max stream plus relation-based aggregation stream

\mathcal{R}:
    dual bag scores

\mathcal{S}:
    slide-level classification label
```

### CTransPath WSI Experiment

```text
\mathcal{G}:
    WSI bag of SRCL-pretrained patch features

\mathcal{C}:
    CLAM-SB attention mechanism

\mathcal{R}:
    attention-weighted slide representation and classifier

\mathcal{S}:
    slide-level classification label
```

The CTransPath backbone is the encoder in this experiment, not the slide
aggregator.

## Family Comparison

| Axis | DSMIL patch pretraining | CTransPath SRCL |
|---|---|---|
| primitive identity | cropped patch | cropped patch |
| guaranteed positive | another view of the same patch | another view of the same patch |
| additional positive | none | top-S target-memory neighbors |
| negative source | other batch views | current minibatch candidates |
| encoder context | ResNet receptive field | CNN stem plus hierarchical shifted-window attention |
| temporal context | none beyond optimization state | EMA target and historical memory bank |
| positive readout | one designated positive logit | log-sum over S+1 positive logits |
| retained object | backbone patch feature | CTransPath patch feature |
| slide context during pretraining | absent | absent |

## Surviving Statistics

### DSMIL SimCLR

The loss preserves a same-patch angular relation against a batch proposal:

```math
\frac{
\exp
\left(
u^{\top}u^{+}/\tau
\right)
}{
\sum_r
\exp
\left(
u^{\top}u_r/\tau
\right)
}.
```

### CTransPath SRCL

The loss preserves total probability of a learned positive set:

```math
\frac{
\sum_{p\in\mathcal{P}_t(z)}
\exp
\left(
z^{\top}p/\tau
\right)
}{
\sum_{p\in\mathcal{P}_t(z)}
\exp
\left(
z^{\top}p/\tau
\right)
+
\sum_n
\exp
\left(
z^{\top}n/\tau
\right)
}.
```

The set itself depends on the current and historical representation geometry.

## Complexity

For a SimCLR batch with `2B` projected views and dimension `d`,
the dense similarity matrix costs:

```math
\Theta
\left(
B^2d
\right).
```

For SRCL with memory size `Q`, mining each of `B` queries by
exhaustive cosine search costs:

```math
\Theta
\left(
BQd
\right),
```

in addition to the minibatch contrast and encoder computation. Exact top-S
selection can be implemented without sorting every memory item, but all
similarities are still evaluated under exhaustive search.

CTransPath window attention for `T` patch-internal tokens costs:

```math
\Theta
\left(
TM^2d_h
\right)
```

per window-attention layer for fixed window size `M`.

## Failure Matrix

| Failure | DSMIL SimCLR | CTransPath SRCL | Location |
|---|---:|---:|---|
| destructive augmentation | yes | yes | positive construction |
| repeated tissue as false negative | yes | reduced only for selected neighbors | candidate graph |
| scanner or stain shortcut | yes | can be amplified by mining | encoder and sampler |
| easiest-positive dominance | no multi-positive set | yes | SRCL readout |
| top-S support switching | no | yes | SRCL graph |
| stale targets | no explicit memory | yes | EMA memory |
| no slide geometry in pretraining | yes | yes | representation boundary |
| frozen feature collision | yes | yes | encoder-to-MIL interface |
| rare-instance dilution | downstream dependent | downstream dependent | slide readout |

## Failure Attribution

Let final prediction be:

```math
\widehat y_i
=
\mathcal{S}_{\mathrm{slide}}
\circ
\mathcal{R}_{\mathrm{slide}}
\circ
\mathcal{C}_{\mathrm{slide}}
\circ
f_{\widehat\theta}^{\otimes n_i}
\left(
\mathcal{B}_i
\right).
```

An error can arise because:

```math
f_{\widehat\theta}
\text{ removed the distinction},
```

because:

```math
\mathcal{C}_{\mathrm{slide}}
\text{ failed to contextualize it},
```

because:

```math
\mathcal{R}_{\mathrm{slide}}
\text{ did not preserve it},
```

or because:

```math
\mathcal{S}_{\mathrm{slide}}
\text{ did not identify it}.
```

## Design Axes Exposed By The Family

The instance family separates four design decisions:

| Axis | Choice in DSMIL | Choice in CTransPath | Open alternative |
|---|---|---|---|
| positive source | exact identity | identity plus nearest memory | spatial, patient, region, or concept relation |
| context scale | convolutional patch context | CNN-Swin patch context | cross-patch or cross-slide context |
| positive readout | one positive | log-sum positive set | equal-weight, robust, or uncertainty-weighted positives |
| slide adaptation | separate MIL head | separate downstream head | end-to-end task-conditioned encoder |

## Bottom Line

DSMIL shows that strong instance features can make weakly supervised slide
learning practical. CTransPath shows that pathology-specific context and mined
cross-instance positives can change the patch geometry. Neither method makes
the patch contrastive loss itself a whole-slide objective.
