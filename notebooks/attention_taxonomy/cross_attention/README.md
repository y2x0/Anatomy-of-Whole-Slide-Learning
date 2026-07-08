# Cross-Attention

Cross-attention uses one object as queries and another object as keys/values.

```math
Q
=
\{q_a\}_{a=1}^{m},
\qquad
K,V
=
\{(k_b,v_b)\}_{b=1}^{n}.
```

This folder studies attention as conditional alignment:

```text
01_queries_keys_values.md
    Q/K/V shapes and conditional measures

02_cross_attention_as_conditional_measure.md
    each query induces a measure over another object

03_multimodal_coattention.md
    pathology-genomics, image-text, and token-token co-attention

04_failure_modes.md
    imported geometry, shortcut alignment, and modality dominance
```

