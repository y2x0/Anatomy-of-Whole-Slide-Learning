# Transformer Attention

Transformer attention applies query-key-value attention to all tokens, often
with multi-head projections, positional structure, and readout tokens.

This folder studies:

```text
01_self_attention_as_complete_graph.md
    self-attention as dense message passing

02_multihead_attention_and_subspaces.md
    heads as parallel learned measures and value subspaces

03_positional_bias_and_geometry.md
    absolute, relative, coordinate, and mask geometry

04_cls_pma_readouts.md
    CLS tokens, PMA seed queries, and slide readout statistics

05_failure_modes.md
    quadratic cost, attention dilution, token bottlenecks, and long-range loss
```

