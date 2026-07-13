# Survival Explanation Targets

Let observed time and event indicator be

```math
X_i=\min(T_i,C_i),
\qquad
\delta_i=\mathbf 1\{T_i\le C_i\}.
```

## 1. Output Objects

```math
\begin{aligned}
\text{Cox:}&\quad \eta_i\in\mathbb R,\\
\text{discrete:}&\quad
\{h_i(\tau_k)\}_{k=1}^K,\\
\text{continuous:}&\quad h_i(t),\\
\text{survival:}&\quad S_i(t),\\
\text{competing risks:}&\quad
\{h_{ic}(t),F_{ic}(t)\}_{c=1}^C.
\end{aligned}
```

An explanation must name which object is targeted.

## 2. Patch-Level Questions

For patch `j`, possible estimands include

```math
\frac{\partial \eta_i}{\partial h_{ij}},
\qquad
\eta_i-\eta_i^{(-j)},
\qquad
S_i(t)-S_i^{(-j)}(t),
\qquad
F_{ic}(t)-F_{ic}^{(-j)}(t).
```

They answer local sensitivity, deletion effect, survival-function effect, and
cause-specific incidence effect. They are not interchangeable.

## 3. Survival Function Direction

For a fixed horizon `t`, larger risk usually means lower survival:

```math
\eta_i\uparrow
\Longrightarrow
S_i(t)\downarrow
```

under a proportional-hazards construction. A heatmap colored “high risk” must
not be relabeled as high instantaneous hazard at every time.

## 4. The Explanation Contract

Every survival heatmap should report target output, horizon, sign convention,
censoring convention, aggregation layer, and whether attention was held fixed
or recomputed under deletion.

