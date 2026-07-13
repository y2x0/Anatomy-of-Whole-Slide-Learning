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

