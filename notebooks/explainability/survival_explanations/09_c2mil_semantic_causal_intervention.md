# C2MIL Semantic Causal Intervention

C2MIL models trivial semantic variation as a confounder of survival prediction.

## 1. Structural Variables

The dual structural causal model distinguishes

```math
T=\text{trivial semantic features},
\qquad
G=\text{graph},
\qquad
C=\text{causal subgraph},
\qquad
S=\text{non-causal subgraph}.
```

The paper's graph includes relations such as

```math
X\leftarrow T\rightarrow Y,
\qquad
X\rightarrow G,
\qquad
S\leftarrow G\rightarrow C.
```

## 2. Backdoor Motivation

If `T` affects both patch representation and outcome through institutional or
staining associations, a naive predictor estimates an association confounded by
`T`. C2MIL uses cross-scale adaptive disentangling to estimate semantic
confounders and construct adjusted features.

The resulting claim is intervention-relative: the model is trained to reduce
dependence on the estimated trivial semantic features. It is not a proof that
all removed variation is clinically irrelevant.

## 3. Survival Target

The adjusted graph feeds the survival objective, typically a Cox-style partial
likelihood in the mapped formulation:

```math
\mathcal L_{\mathrm{Cox}}
=-\sum_{i:\delta_i=1}
\left(
\eta_i-\log\sum_{r\in\mathcal R_i}\exp\eta_r
\right).
```

Semantic explanations must therefore name whether they target adjusted risk,
unadjusted risk, or the survival curve derived from the adjusted risk.

