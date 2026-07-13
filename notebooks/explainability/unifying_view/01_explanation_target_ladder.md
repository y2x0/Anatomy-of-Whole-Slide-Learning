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

