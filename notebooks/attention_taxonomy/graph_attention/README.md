# Graph Attention

Graph attention restricts attention to edges.

```math
h_u'
=
\sum_{v\in\mathcal{N}(u)}
a_{uv}W h_v.
```

This folder studies attention as learned edge weighting:

```text
01_attention_on_edges.md
    attention coefficients over graph neighborhoods

02_attention_vs_adjacency_learning.md
    fixed adjacency, learned weights, and learned topology

03_oversmoothing_oversquashing_attention.md
    attention under message-passing depth and graph bottlenecks

04_failure_modes.md
    wrong edges, attention shortcuts, degree bias, and topology errors
```
