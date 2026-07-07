# Unifying View

This folder places foundation model adaptation inside the repository-wide
C/R/G/S framework.

The core abstraction is:

```math
\widehat y
=
T_{\phi_0,\eta}(X,S),
```

where $\phi_0$ is pretrained and $\eta$ is the adaptation state.

## Files

- `01_task_information_injection.md`: where labels change the computation.
- `02_parameter_partition_and_gradient_flow.md`: frozen versus trainable paths.
- `03_crgs_adaptation_matrix.md`: adaptation families in C/R/G/S form.
- `04_paper_placement_matrix.md`: pathology FM papers and adaptation modes.
- `05_decision_checklist.md`: how to report an adaptation method clearly.

## Dense Summary

Foundation adaptation is not one method family. It is a question about the
location and dimension of downstream task information.
