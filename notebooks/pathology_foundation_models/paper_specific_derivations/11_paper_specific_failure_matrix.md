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
