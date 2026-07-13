# Competing Risks

Competing risks appear when more than one mutually exclusive event type can
terminate observation.

For patient
```math
i
```
:

```math
T_i=\min_{c\in\{1,\ldots,C\}}T_{ic},
\qquad
J_i=\arg\min_{c}T_{ic}.
```

The observed data are:

```math
X_i=\min(T_i,C_i),
\qquad
\delta_i=\mathbf{1}[T_i\le C_i],
\qquad
\Delta_i=\delta_iJ_i.
```

Here
```math
\Delta_i=0
```
 means censoring, and
```math
\Delta_i=c
```
 means event type
```math
c
```
.

The core question is:

```text
Does the model represent cause-specific instantaneous risk, or cumulative
incidence for each event type?
```

These are not equivalent objects.

## Files

- `01_data_cif_and_event_objects.md`: event labels, CIFs, and survival.
- `02_cause_specific_hazards.md`: cause-specific hazards and likelihood.
- `03_subdistribution_hazards_fine_gray.md`: Fine-Gray and CIF regression.
- `04_neural_competing_risks_and_deephit.md`: PMF models and DeepHit-style loss.
- `05_wsi_competing_risk_design.md`: WSI heads for multiple event types.
- `06_failure_modes.md`: interpretability and censoring traps.

## Anchor Papers

- Fine and Gray. "A proportional hazards model for the subdistribution of a
  competing risk." JASA 1999.
- Lee et al. "DeepHit: A Deep Learning Approach to Survival Analysis With
  Competing Risks." AAAI 2018.
  https://ojs.aaai.org/index.php/AAAI/article/view/11842
- Dynamic-DeepHit. IEEE T-BME 2020.
  https://pubmed.ncbi.nlm.nih.gov/30951460/
