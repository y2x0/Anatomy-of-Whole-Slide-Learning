# Time-Dependent AUC

Time-dependent AUC evaluates discrimination at a time horizon.

Let $M_i(t)$ be a marker or predicted risk at time $t$. Commonly:

```math
M_i(t)=1-\widehat{S}_i(t).
```

## Cumulative/Dynamic Definition

At horizon $t$, cumulative cases are:

```math
\mathcal{C}(t)=\{i:T_i\le t\}.
```

Dynamic controls are:

```math
\mathcal{D}(t)=\{i:T_i>t\}.
```

Sensitivity at threshold $a$:

```math
\operatorname{TPR}(a,t)
=
\Pr(M(t)>a\mid T\le t).
```

Specificity:

```math
\operatorname{TNR}(a,t)
=
\Pr(M(t)\le a\mid T>t).
```

The cumulative/dynamic AUC is:

```math
\operatorname{AUC}_{\operatorname{CD}}(t)
=
\Pr(M_i(t)>M_j(t)\mid T_i\le t,T_j>t).
```

## Incident/Dynamic Definition

Incident cases are subjects failing near time $t$:

```math
T_i\in[t,t+dt).
```

Dynamic controls still satisfy:

```math
T_j>t.
```

The incident/dynamic AUC asks whether the marker separates subjects failing now
from subjects still at risk.

## Censoring

Both definitions require censoring adjustment. IPCW estimators weight observed
case/control contributions by censoring survival:

```math
\widehat{G}(u)=\Pr(C>u).
```

The exact weights depend on the AUC definition.

## Relation To C-Index

C-index summarizes ordering over event times. Time-dependent AUC gives a curve:

```math
t\mapsto \operatorname{AUC}(t).
```

A model can discriminate well early but poorly late:

```math
\operatorname{AUC}(1\text{ year}) >
\operatorname{AUC}(5\text{ years}).
```

This matters for WSI because morphology may encode early aggressiveness better
than long-term treatment response.

## Risk Functional

For survival curves:

```math
M_i(t)=1-\widehat{S}_i(t)
```

is natural. For Cox:

```math
M_i(t)=\eta_i
```

is time-independent. Cox can still have time-varying AUC because the case/control
sets change with $t$.

## WSI Use

For WSI models:

```math
S_i
\to
H_i
\to
z_i
\to
\widehat{S}_i(t).
```

Time-dependent AUC can show which horizons are actually supported by slide
morphology.

If a paper reports only one C-index, it hides this horizon dependence.

## Dense Summary

```math
\operatorname{AUC}_{\operatorname{CD}}(t)
=
\Pr(M_i(t)>M_j(t)\mid T_i\le t,T_j>t).
```

Time-dependent AUC evaluates discrimination at a clinically specified horizon,
not global ranking over all follow-up.
