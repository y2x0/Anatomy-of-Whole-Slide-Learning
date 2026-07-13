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

## Algebra Of The Toy Tests

The tests can be stated as equality or inequality constraints on the forward map.
For two slides with identical patch features but different layouts,

```math
H=H',\quad G\ne G'
\quad\Longrightarrow\quad
\widehat y(H,G)\ne\widehat y(H',G')
```

is evidence that geometry enters the computation; equality is evidence of layout
invariance, not necessarily of geometric understanding. For a heatmap h and a
model score q, two explanations can be non-identifiable when

```math
q(F(X))=q(F(X'))
\quad\text{while}\quad
h(X)\ne h(X').
```

For a graph, deleting a bridge node b tests topology dependence through

```math
\Delta_b
=
q(F(G,X))-q(F(G\setminus b,X_{\setminus b})).
```

For survival, equal ranking does not imply equal calibration:

```math
\eta_i>\eta_j
\quad\text{and}\quad
\widetilde S_i(\tau)>\widetilde S_j(\tau)
```

can hold for two models with different absolute horizon probabilities. The test
must therefore compare both ordering and calibrated output at the reported
horizon.
