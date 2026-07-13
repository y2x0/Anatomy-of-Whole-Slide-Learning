# Explanation Failure Matrix and Sanity Checks

| Failure | Hidden assumption | Sanity check |
|---|---|---|
| attention equals evidence | routing is signed score credit | weight-preserving score test |
| smooth map equals spatial reasoning | coordinates entered the computation | same features, permuted layout |
| prototype equals morphology | metric neighborhood is semantic | expert purity and prompt/stain shifts |
| concept equals cause | representation direction is intervention-valid | finite edit and external shift |
| deletion equals necessity | removed object is replaced feasibly | delete/recompute versus fixed context |
| graph edge equals interaction | edge weight is effect, not normalization | edge intervention |
| risk equals survival | Cox score is treated as a curve | baseline hazard and calibration |
| co-attention equals mechanism | cross-modal routing is causal | token-specific ablation |
| causal subgraph is true cause | SCM and invariance are correct | held-out domain intervention |

## Minimal Toy Tests

```math
\begin{aligned}
\text{same set, different layout}&\to\text{tests geometry},\\
\text{same score, different heatmap}&\to\text{tests identifiability},\\
\text{same map, different head}&\to\text{tests target dependence},\\
\text{delete one bridge node}&\to\text{tests topology brittleness},\\
\text{same risk, different calibration}&\to\text{tests survival output semantics}.
\end{aligned}
```

