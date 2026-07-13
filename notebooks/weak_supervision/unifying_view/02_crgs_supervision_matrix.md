# C/R/G/S Supervision Matrix

Weak supervision begins with an observed signal:

```math
S_i^{\mathrm{obs}}
\sim
Q_\alpha(S\mid U_i,H_i,G_i).
```

Some methods then create generated targets:

```math
\widehat U_{i,t}
=
\Psi_t(H_i,G_i,S_i^{\mathrm{obs}},\theta_t,\mathcal{D}).
```

Both can enter the C/R/G/S pipeline:

```math
\widetilde H_i
=
\mathcal{C}(H_i;G_i,S_i^{\mathrm{obs}},\widehat U_{i,t}),
\qquad
z_i
=
\mathcal{R}(\widetilde H_i;G_i,S_i^{\mathrm{obs}},\widehat U_{i,t}),
\qquad
\widehat y_i
=
\mathcal{H}(z_i).
```

The same observed label can affect different parts of the pipeline.

## Matrix

| Supervision | Observed S^{\mathrm{obs}} | Generated target | \mathcal{C} | \mathcal{R} | G | Main latent variable |
|---|---|---|---|---|---|---|
| Bag label | Y_i | none unless auxiliary rule added | instance features shaped only through bag loss | MIL aggregator approximates bag map | optional | Z_{ij} |
| Partial label | M\odot U | imputed labels under a mask if used | direct gradients on observed subset | bag and partial readouts can be joint | annotation geometry | unobserved labels under mask |
| Noisy label | \widetilde Y_i | corrected posterior if noise model is inverted | features may fit corruption | readout predicts noisy or corrected label | possible noise correlate | true Y_i |
| Pseudo-label | usually Y_i or partial labels | \widehat U_t=\Psi_t(H,G,S^{\mathrm{obs}},\theta_t,\mathcal{D}) | generated labels shape features | selects top-k, pseudo-bags, or teacher targets | optional | correctness of pseudo-label |
| Contrastive relation | \mathcal{P},\mathcal{N} or paired views/text | sampled positives and negatives | pairwise invariance shapes encoder | embedding object is contrasted | relation geometry | latent equivalence relation |

## S In Context

Supervision enters context when it shapes instance features:

```math
\widetilde h_{ij}
=
\mathcal{C}_\theta(h_{ij};S_i^{\mathrm{obs}},\widehat U_{i,t}).
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
\mathcal{R}_\theta(H_i;S_i^{\mathrm{obs}},\widehat U_{i,t}).
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
G(H_i,S_i^{\mathrm{obs}},\widehat U_{i,t}).
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
\ell(\widehat y_i,S_i^{\mathrm{obs}}).
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
