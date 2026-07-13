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

## Target-Specific Explanation Functionals

Let I_a remove, mask, or otherwise intervene on patch or node a, with the full
context and readout recomputed. For a Cox score, the model-relative explanation is

```math
\Delta_{\eta}(a)
=
\eta_i(H_i)-\eta_i(\mathsf I_aH_i).
```

For a discrete hazard at horizon k, the same intervention has a time-indexed
effect:

```math
\Delta_{h,k}(a)
=
h_i(\tau_k;H_i)-h_i(\tau_k;\mathsf I_aH_i).
```

For a survival curve, a scalar explanation requires an explicitly chosen
functional, for example the integrated absolute curve change:

```math
\Delta_{S,[0,\tau]}(a)
=
\int_0^{\tau}
\left|
S_i(t;H_i)-S_i(t;\mathsf I_aH_i)
\right|\,d\mu(t).
```

For competing risks, explain the cause-specific incidence rather than silently
using all-cause survival:

```math
\Delta_{F,c,k}(a)
=
F_{ic}(\tau_k;H_i)-F_{ic}(\tau_k;\mathsf I_aH_i).
```

These quantities can disagree in sign. A patch can increase one cause-specific
incidence while decreasing another, or change early hazard while leaving a late
survival summary nearly unchanged. One heatmap cannot represent all of these
targets without an explicit aggregation rule.

## Censoring And Explanation Scope

The intervention effect above is a change in the fitted model output. Censoring
enters validation through the cohort-level metric, not by changing the definition
of the per-slide model delta. A reported explanation should therefore separate

```text
model delta:
    change in the predicted target after intervention

validation target:
    censoring-aware agreement with observed outcomes

biological claim:
    effect under a justified structural intervention
```

Conflating these levels turns a faithful computational perturbation into an
unsupported biological explanation.
