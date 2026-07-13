# Competing Risk and Cause-Specific Explanations

## 1. Cause-Specific Hazards

For causes `c=1,...,C`,

```math
h_{ic}(t)
=\lim_{\Delta t\to0}
\frac{P(t\le T_i<t+\Delta t, J_i=c\mid T_i\ge t)}{\Delta t}.
```

The all-cause survival function is

```math
S_i(t)
=\exp\left[-\int_0^t\sum_c h_{ic}(u)\,du\right].
```

## 2. Cumulative Incidence

Cause-specific cumulative incidence is

```math
F_{ic}(t)
=\int_0^t S_i(u)h_{ic}(u)\,du.
```

An explanation for cause `c` must specify whether it targets `h_ic`, `F_ic`, or
the all-cause survival function.

## 3. Shared Versus Cause-Specific Attention

Shared attention produces one statistic `z_i` for all causes. Cause-specific
attention produces

```math
z_{ic}=\sum_j a_{ijc}r_{ij}.
```

Shared attention cannot by itself explain why two causes differ. Cause-specific
maps can overlap or compete, and their weights need not sum across causes.

## 4. Cause Interaction

A patch may increase one cause hazard and decrease another. The net survival
effect is mediated by the sum of cause hazards, while incidence effects also
depend on event-free survival. A heatmap called “risk” is incomplete without the
event definition.

## 5. Censoring and Competing Events

Deleting a competing event from an analysis is not a neutral explanation. A
counterfactual cause-specific quantity requires a specified event process and
cannot be read from an observed label as if other causes were absent.

