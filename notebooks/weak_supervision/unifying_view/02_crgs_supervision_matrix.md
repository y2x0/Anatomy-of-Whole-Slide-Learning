# C/R/G/S Supervision Matrix

Weak supervision is the $S$ in:

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

The same observed label can affect different parts of the pipeline.

## Matrix

| Supervision | $S$ | $\mathcal{C}$ | $\mathcal{R}$ | $G$ | Main Latent Variable |
|---|---|---|---|---|---|
| Bag label | $Y_i$ | instance features shaped only through bag loss | MIL aggregator approximates bag map | optional | $Z_{ij}$ |
| Partial label | $M\odot U$ | direct gradients on observed subset | bag and partial readouts can be joint | annotation geometry | unobserved labels under mask |
| Noisy label | $\widetilde Y_i$ | features may fit corruption | readout predicts noisy or corrected label | possible noise correlate | true $Y_i$ |
| Pseudo-label | $\widehat Z$ or $\widehat Y$ | generated labels shape features | selects top-k, pseudo-bags, or teacher targets | optional | correctness of pseudo-label |
| Contrastive relation | $\mathcal{P},\mathcal{N}$ | positive-pair invariance shapes encoder | embedding object is contrasted | relation geometry | latent equivalence relation |

## S In Context

Supervision enters context when it shapes instance features:

```math
\widetilde h_{ij}
=
\mathcal{C}_\theta(h_{ij};S_i).
```

Examples:

```text
pseudo-instance labels:
    direct instance classification loss

contrastive pairs:
    representation geometry from positives and negatives

teacher-student:
    teacher targets shape student features
```

## S In Readout

Supervision enters readout when it decides which instances are selected or
weighted:

```math
z_i
=
\mathcal{R}_\theta(H_i;S_i).
```

Examples:

```text
CLAM:
    slide label plus attention extremes create top/bottom-k pseudo-labels

DTFD-MIL:
    pseudo-bag labels shape local readouts

noisy-or:
    bag label and OR assumption define probabilistic readout
```

## S In Geometry

Supervision enters geometry when labels define neighborhoods or topology:

```math
G_i
=
G(H_i,S_i).
```

Examples:

```text
contrastive positives:
    same-label slides become neighbors

pseudo-bag construction:
    clustering or attention partitions create regions

learned graph:
    label gradients shape task-dependent topology
```

## S In Task Head

The weakest use of supervision is only at the final head:

```math
\widehat y_i
=
\mathcal{H}(z_i),
\qquad
\mathcal{L}
=
\ell(\widehat y_i,S_i).
```

ABMIL-style bag classification often lives here unless auxiliary losses are
added.

## Dense Summary

Weak supervision is not just the final label. It can shape:

```text
C:
    feature geometry

R:
    instance selection and aggregation

G:
    relation structure

H:
    task prediction
```

The method should state where the supervision enters.
