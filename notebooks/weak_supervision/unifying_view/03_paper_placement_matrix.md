# Paper Placement Matrix

This matrix places anchor methods by the supervision channel they use.

## Matrix

| Method or family | Observed $S$ | Latent $U$ | Channel assumption | Loss shape | Main failure |
|---|---|---|---|---|---|
| Campanella-style clinical weak supervision | report-derived slide label | true diagnosis and latent diagnostic regions | report label approximates slide truth | slide CE over large cohorts | report noise and shortcut witnesses |
| ABMIL | slide label | latent positive instances and attention evidence | bag label supervises weighted slide statistic | bag CE | attention not identifiable as patch truth |
| CLAM | slide label plus generated top/bottom-k pseudo-labels | latent class-specific instance labels | attention extremes approximate positives/negatives | slide CE plus instance CE | self-reinforcing pseudo-label errors |
| DSMIL | slide label plus self-supervised representation signal | latent critical instance and related morphology | max witness plus critical-instance neighborhood | instance max branch plus bag branch | false critical instance amplification |
| DTFD-MIL | slide label converted to pseudo-bag/distilled targets | latent informative pseudo-bags | pseudo-bag partitions concentrate signal | local/global distillation | pseudo-bag label noise |
| Cluster-to-Conquer | clusters plus slide labels | latent morphology groups | cluster structure helps localize evidence | cluster-aware MIL losses | cluster-label mismatch |
| Weakly semi-supervised WSI | few labels plus unlabeled/consistency signals | missing slide or instance labels | augmentations preserve targets | supervised plus consistency | biased unlabeled pool |
| SC-MIL / supervised contrastive MIL | class labels define positives | latent class morphology relation | same class should be close | CE plus supervised contrastive | same-label heterogeneity |
| SCL-WC / cross-slide contrast | cross-slide positives | latent shared morphology | selected slides share pathology-relevant signal | contrastive objective | false positive and false negative pairs |
| LACL | lesion-aware positives | latent lesion evidence | lesion-aware relation improves representation | contrastive and task losses | lesion heuristic error |

## Campanella-Style Clinical Weak Supervision

The observed signal is a slide-level label derived from clinical diagnosis or
report metadata:

```math
S_i
=
\widetilde Y_i.
```

The method relies on scale:

```math
N
\text{ large enough that noisy weak labels still train a useful slide classifier}.
```

The latent instance explanation remains weakly identified.

## ABMIL

ABMIL observes:

```math
S_i
=
Y_i.
```

It learns:

```math
z_i
=
\sum_j a_{ij}v(h_{ij}),
\qquad
\widehat y_i
=
\mathcal{H}(z_i).
```

The supervision channel constrains $z_i$ to predict $Y_i$, not $a_{ij}$ to equal
instance truth.

## CLAM

CLAM expands supervision:

```math
S_i
=
\left(
Y_i,
\widehat Z_i^{\mathrm{top/bottom}}
\right).
```

where:

```math
\widehat Z_i^{\mathrm{top/bottom}}
=
\Psi(a_i,Y_i).
```

Its mathematical bet is:

```math
\Psi(a_i,Y_i)
\approx
\text{true instance labels on selected extremes}.
```

## DSMIL

DSMIL uses a latent critical witness:

```math
j_i^\star(c)
=
\arg\max_j s_{ijc}.
```

The bag branch then uses this witness as a query:

```math
a_{ij}^{(c)}
\propto
\exp(q_{i,j_i^\star(c)}^\top q_{ij}).
```

The supervision channel is still slide-level, but the architecture imposes a
latent critical-instance explanation.

## DTFD-MIL And Pseudo-Bags

Pseudo-bag methods create:

```math
\widehat S_i
=
\{(B_{im},\widehat Y_{im})\}_{m=1}^{M_i}.
```

This converts one huge weak problem into smaller weak problems. The key
assumption is:

```math
P(\widehat Y_{im}=Y_{im}^{\mathrm{true}})
```

is high enough to improve learning.

## Contrastive WSI Methods

Contrastive methods observe relation labels:

```math
S
=
\mathcal{P}
```

where $\mathcal{P}$ is a positive-pair set. The latent assumption is:

```math
(a,b)\in\mathcal{P}
\quad\Longrightarrow\quad
U_a\sim U_b.
```

The failure mode is that the pair relation may be only a weak proxy for shared
pathology.

## Dense Summary

The same paper can use multiple supervision channels:

```text
slide CE:
    bag label

instance auxiliary loss:
    pseudo-label

pretraining:
    contrastive or self-supervised relation

distillation:
    teacher-generated target
```

The matrix should be read by channel, not by architecture name.
