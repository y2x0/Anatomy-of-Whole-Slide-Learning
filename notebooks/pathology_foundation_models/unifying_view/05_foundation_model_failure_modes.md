# Foundation Model Failure Modes

| Failure | Cause | Blunt diagnostic |
|---|---|---|
| augmentation erases lesion | positive view changes pathology | lesion-preserving augmentation audit |
| negative false contrast | negative shares patient or tissue state | patient-aware negative sampling |
| cluster reinforcement | pseudo-clusters encode scanner | external cluster purity |
| teacher blind spot | student copies unvalidated teacher | teacher-student subgroup gaps |
| masked shortcut | reconstruction uses stain/context only | stain and morphology perturbations |
| caption shortcut | text reveals institution or label | prompt and report ablation |
| slide bottleneck | region pooling discards rare patch | rare-region recall |
| retrieval cohort bias | memory lacks external morphology | cross-cohort nearest neighbors |
| knowledge prior error | injected concepts are incomplete/wrong | knowledge ablation |
| adaptation mismatch | downstream head cannot access signal | compare probe, MIL, and fine-tune |

## Core Principle

Foundation-model scale can reduce variance while increasing confidence in a
wrong inductive bias. The audit must test the bias, not only the average score.

## Contract-Level Failure Equations

Let a patch or slide encoder be

```math
z=f_{\phi}(x).
```

Every pretraining objective imposes a contract on which distinctions in x should
survive in z. A failure occurs when the contract is satisfied while a downstream
task-relevant distinction is removed.

### Augmentation Invariance

Contrastive and self-distillation objectives often require two views to agree:

```math
f_{\phi}(t_1x)\approx f_{\phi}(t_2x).
```

If an augmentation changes a lesion or a stain pattern that the downstream task
needs, the learned equivalence class is too large:

```math
t_1x\not\sim_{\mathrm{task}}t_2x
\quad\text{but}\quad
f_{\phi}(t_1x)\approx f_{\phi}(t_2x).
```

The correct audit is not only augmentation invariance. It is invariance measured
against labels or expert morphology under lesion-preserving and lesion-altering
transformations.

### Negative Sampling And Clustering

For a contrastive loss, let Q be the negative-pair distribution. The learned
geometry is shaped by Q, not by the name of the loss:

```math
\mathcal L_{\mathrm{NCE}}
=
-\log
\frac{\exp(\mathrm{sim}(z_i,z_i^+)/\tau)}
{\exp(\mathrm{sim}(z_i,z_i^+)/\tau)
 +\sum_{j\sim Q}\exp(\mathrm{sim}(z_i,z_j)/\tau)}.
```

If Q contains same-patient, same-lesion, or same-morphology negatives, the loss
penalizes clinically meaningful similarity. Cluster-guided retrieval adds another
random variable:

```math
c_i=\kappa(f_{\phi}(x_i)),
\qquad
Q=Q_{\kappa(f_{\phi})}.
```

An error in the cluster map therefore changes the positive and negative laws and
can reinforce itself during subsequent training.

### Teacher-Student And Distillation

A teacher-student objective constrains the student toward a moving target:

```math
\mathcal L_{\mathrm{distill}}
=
D\left(
q_{\bar\phi}(t_2x),
p_{\phi}(t_1x)
\right),
\qquad
\bar\phi\leftarrow m\bar\phi+(1-m)\phi.
```

The student cannot recover a teacher-invariant distinction without an additional
loss or view. A subgroup gap between teacher and student is therefore evidence of
compressed error, not evidence that the student has learned a new pathology
signal.

### Masked Reconstruction

Masked modeling estimates a conditional reconstruction:

```math
\widehat x_M
=
g_{\psi}\left(f_{\phi}(x_{\setminus M}),M\right),
\qquad
\mathcal L_M
=
\mathbb E\left[d(\widehat x_M,x_M)\right].
```

The optimum preserves information useful for predicting the masked target, not
necessarily information useful for diagnosis. If stain, tissue layout, or scanner
context predicts x_M more cheaply than morphology, reconstruction can succeed
while task-relevant morphology is underrepresented.

### Image-Text Alignment

An image-text model contracts visual and textual embeddings:

```math
\mathcal L_{\mathrm{align}}
=
-\log
\frac{\exp(\mathrm{sim}(v_i,t_i)/\tau)}
{\sum_j\exp(\mathrm{sim}(v_i,t_j)/\tau)}.
```

The learned visual statistic is sufficient for the text distribution only under
the dataset's reporting channel. It need not identify a local lesion or a causal
slide feature. Negation, paraphrase, patient identity, and institution metadata
are direct tests of whether the model learned morphology or caption shortcuts.

### Slide Bottlenecks And Retrieval

A slide foundation encoder still has a C/R map:

```math
z_i
=
\mathcal R_{\theta}
\left(
\mathcal C_{\theta}(H_i;G_i)
\right).
```

If two slides differ only in a rare patch and the map gives the same z, every
downstream probe sees them as equivalent. Retrieval makes the archive part of the
prediction object:

```math
\mathcal N_K(i)
=
\underset{A\subset\mathcal M,\,|A|=K}{\arg\min}
\sum_{a\in A}d(z_i,z_a).
```

Changing the cohort changes the memory M, the neighbors, and potentially the
prediction even when the encoder is frozen. A retrieval explanation must therefore
report the archive, distance, exclusion rules, and external-cohort behavior.

### Knowledge And Adaptation

Knowledge-enhanced alignment introduces a prior or transformed text target:

```math
t_i'=\Psi(t_i,\mathcal K),
\qquad
\mathcal L_{\mathrm{V\text{-}L}}
=
\mathcal L(v_i,t_i').
```

The prior can improve semantic coverage while importing errors from the knowledge
source. Finally, a frozen encoder plus a readout optimizes only

```math
\min_{\psi}\mathcal L\left(\mathcal H_{\psi}(\mathcal R(f_{\phi_0}(H)))\right),
```

whereas fine-tuning also moves phi. A failed linear probe is evidence about the
adaptation boundary, not necessarily evidence that the foundation representation
contains no useful signal.
