# Perturbation Faithfulness And C/R/G/S

## Deletion And Insertion Curves

Given ranking `i_1,...,i_n`, define:

```math
P_k^{-}
=
P
\setminus
\left\{
p_{i_1},\ldots,p_{i_k}
\right\},
```

```math
D(k)
=
F(P_k^{-}).
```

For baseline bag `P_0`:

```math
P_k^{+}
=
P_0
\cup
\left\{
p_{i_1},\ldots,p_{i_k}
\right\},
```

```math
I(k)
=
F(P_k^{+}).
```

Deletion favors rapid decrease; insertion favors rapid increase. Different
paths and baselines can reverse method rankings.

## Random-Control Curve

Compare explanation ordering to random permutations:

```math
\Delta_{\mathrm{faith}}
=
\mathrm{AUC}_{\mathrm{random}}
-
\mathrm{AUC}_{\mathrm{explanation}}.
```

This tests improvement over chance under the same perturbation protocol.

## C/R/G/S Matrix

| Method | Context `C` | Readout `R` | Geometry `G` | Explanation target `S` |
|---|---|---|---|---|
| leave-one-out | full recomputed model | one removed instance | finite score difference | single-patch necessity |
| LIME | sampled masked inputs | sparse local linear coefficients | kernel-weighted regression | local surrogate behavior |
| SHAP | coalition game | additive Shapley allocation | average marginal contribution | game-relative score decomposition |
| HIPPO-knowledge | set intervention selected by annotation | region removal or addition | counterfactual model response | hypothesis-driven necessity or sufficiency |
| HIPPO-attention | attention-selected set intervention | top-attention ablation | score response | empirical test of attention influence |
| HIPPO-search | sequential leave-one-out recomputation | greedy selected subset | context-updated marginal effects | strong positive or negative model drivers |

## Failure Matrix

| Failure | Mathematical source |
|---|---|
| cardinality artifact | deletion changes set size and normalization |
| off-manifold replacement | baseline tissue or embedding is unrealistic |
| interaction double counting | singleton effects overlap |
| surrogate instability | sampled mask design is ill-conditioned |
| coalition ambiguity | marginal, conditional, and deletion games differ |
| redundant evidence blindness | one-at-a-time removal has small effect |
| greedy suboptimality | non-submodular set objective |
| biological overclaim | model intervention is interpreted as patient-level causality |

## Unified Failure Principle

```math
\text{player or region definition}
\longrightarrow
\text{removal, replacement, or addition rule}
\longrightarrow
\text{model recomputation}
\longrightarrow
\text{score contrast}
\longrightarrow
\text{attribution or ranking}.
```

Perturbation explanations are unusually concrete because they measure finite
model behavior. Their meaning remains inseparable from the intervention that
created the counterfactual input.
