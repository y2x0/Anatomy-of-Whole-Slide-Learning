# Paper-Specific Foundation Model Failure Matrix

| Model family | Preserved statistic | Main failure | Audit |
|---|---|---|---|
| CTransPath | augmentation-invariant patch geometry | wrong positive invariance | pathology-aware augmentations |
| RetCCL | cluster-guided retrieval neighborhood | cluster error reinforcement | independent retrieval labels |
| HIPT | hierarchical local-to-global relation | region bottleneck | scale and partition ablation |
| UNI/Virchow | broad patch morphology | rare or institution-specific signal | external cohort and rare-class probes |
| CONCH/PLIP | image-text semantic alignment | caption or prompt shortcut | paraphrase, negation, expert concept tests |
| Prov-GigaPath | long-context slide statistic | missing-tile and coordinate dependence | tessellation and layout stress tests |
| TITAN | slide-level semantic/retrieval geometry | report and retrieval cohort bias | external retrieval and morphology audit |
| KEEP | knowledge-enhanced visual-language geometry | knowledge prior error | knowledge ablation and held-out concepts |
| distillation | teacher function or relation | teacher blind spots compressed | teacher-student subgroup comparison |

The model is not the explanation. The preserved statistic and its failure mode
are the explanation-relevant units.

## Common Audit Map

Each row can be written as a contract on a learned map:

```math
z=f_{\phi}(x),
\qquad
\mathcal E_{\mathrm{pre}}(z;\text{views},\text{pairs},\text{targets})
\longrightarrow
\text{task-relevant equivalence classes}.
```

The audit asks whether the stated preserved statistic survives the corresponding
intervention. For an augmentation-based model such as CTransPath or HIPT,

```math
\Delta_{\mathrm{view}}
=
d\left(f_{\phi}(t_1x),f_{\phi}(t_2x)\right)
```

should be small for declared-equivalent views but not necessarily for a lesion-
altering transformation. For RetCCL, the retrieval neighborhood is a function of
the learned cluster map:

```math
\mathcal N_K(x)
=
\underset{A\subset\mathcal M,\,|A|=K}{\arg\min}
\sum_{a\in A}d(f_{\phi}(x),f_{\phi}(a)),
\qquad
c(x)=\kappa(f_{\phi}(x)).
```

Changing the archive or cluster assignments is therefore a direct test of whether
the retrieval claim is stable or circular.

For slide-level models, define the representation equivalence relation

```math
S\sim S'
\quad\Longleftrightarrow\quad
\mathcal R(\mathcal C(S;G))
=
\mathcal R(\mathcal C(S';G')).
```

The bottleneck audits for HIPT, Prov-GigaPath, and TITAN should construct pairs
that differ in one region, scale, coordinate arrangement, or report context and
measure whether the claimed slide statistic changes. For distillation and
knowledge-enhanced alignment, the corresponding tests compare the student or
knowledge-conditioned output against the teacher and against held-out concepts:

```math
\Delta_{\mathrm{teacher}}
=
d\left(f_{\mathrm{student}}(x),f_{\mathrm{teacher}}(x)\right),
\qquad
\Delta_{\mathrm{knowledge}}
=
\mathcal L(x,\Psi(t,\mathcal K))
-
\mathcal L(x,t).
```

The matrix is thus a list of falsifiable interventions, not a list of generic
failure words. Each paper-specific audit should name the perturbation, the
preserved statistic, and the output that is expected to change.
