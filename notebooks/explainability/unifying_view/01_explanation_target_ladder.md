# Explanation Target Ladder

Let the full WSI pipeline be

```math
X_i
\xrightarrow{\Phi}
H_i
\xrightarrow{\mathcal C}
\widetilde H_i
\xrightarrow{\mathcal R}
z_i
\xrightarrow{\mathcal H}
F_i.
```

An explanation can target any level:

```math
\begin{aligned}
\text{input:}&\quad X_i,\\
\text{representation:}&\quad H_i,\\
\text{context:}&\quad \widetilde H_i,\\
\text{readout:}&\quad z_i,\\
\text{decision:}&\quad F_i.
\end{aligned}
```

## Claim Ladder

```text
presence      -> an object is detected
routing       -> an object receives computational weight
score credit  -> an object contributes to a chosen scalar output
necessity     -> removing it changes that output
sufficiency   -> retaining it can recover that output
causality     -> an intervention represents a valid structural action
```

Each step adds assumptions. A heatmap that establishes presence cannot be
reported as necessity or causality without an additional intervention.

## Formal Explanation Functional

Let q be the scalar quantity being explained. It may be a class logit, a Cox
risk score, a survival probability at a horizon, or a coordinate of the learned
representation. An explanation is a functional

```math
E_q(X_i;\Phi,\mathcal C,\mathcal R,\mathcal H)
\in
\mathcal A,
```

where A is the attribution space: patches, nodes, edges, pixels, concepts, or
interventions. The target must be fixed before the attribution is interpreted:

```math
q(F_i)=
\begin{cases}
F_{ic}, & \text{class score},\\
\eta_i, & \text{scalar survival risk},\\
S_i(\tau), & \text{survival at horizon }\tau,\\
F_{ic}(\tau), & \text{cause-specific incidence},\\
z_{ir}, & \text{representation coordinate}.
\end{cases}
```

For an intervention I_a that modifies object a, necessity is a statement about
the model effect

```math
\Delta_q(a)
=
q(F_i(X_i))-q(F_i(\mathsf I_a(X_i))).
```

Presence, routing, score credit, necessity, and causality are therefore not
different color scales for one quantity. They are different claims about E, q,
and the admissible intervention family.
