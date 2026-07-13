# Survival Explanation Validation and C/R/G/S

## 1. Validation Matrix

| Target | Explanation | Required validation |
|---|---|---|
| Cox risk `eta` | signed patch or node credit | score completeness, deletion effect, concordance target |
| hazard `h(t)` | time-specific map | horizon perturbation and hazard calibration |
| survival `S(t)` | curve effect | integrated curve difference and calibration |
| incidence `F_c(t)` | cause-specific map | event-type definition and competing-risk evaluation |
| co-attention | modality-conditioned routing | target-specific ablation and token stability |
| causal subgraph | semantic/topological intervention | invariance, fidelity, sparsity, external shift |

## 2. Metric Mismatch

High C-index on `eta` does not prove calibrated `S(t)`. A patch explanation
faithful to risk ordering can fail to explain horizon calibration. Report the
metric that matches the explained object.

## 3. C/R/G/S Placement

```math
H_i
\xrightarrow{\mathcal C(G,S)}
\widetilde H_i
\xrightarrow{\mathcal R}
z_i\text{ or }z_i(t,c)
\xrightarrow{\mathcal H}
\eta_i,h_i(t),S_i(t),F_{ic}(t).
```

| Family | Context | Readout | Surviving statistic | Main explanation |
|---|---|---|---|---|
| Cox MIL | attention or graph | weighted sum | relative-risk score | signed risk credit |
| discrete MIL | attention, transformer, or graph | bin-specific head | hazard vector | horizon-specific credit |
| continuous MIL | time-conditioned context | function-valued readout | hazard curve | time-functional effect |
| MCAT | genomic-guided co-attention | multimodal transformer | fused risk representation | gene-conditioned patch map |
| C2MIL | semantic adjustment and causal topology | selected graph readout | adjusted risk and causal subgraph | intervention-relative subgraph |

## 4. Minimum Reporting Rule

Every survival explanation should state target function, time horizon or event
type, censoring convention, baseline, signed direction, whether context was
recomputed, and whether the claim is associational, computationally causal, or
biological.

