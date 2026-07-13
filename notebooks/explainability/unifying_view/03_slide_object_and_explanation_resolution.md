# Slide Object and Explanation Resolution

## 1. Slide Objects

```math
\begin{aligned}
\text{set:}&\quad H_i=\{h_{ij}\}_j,\\
\text{sequence:}&\quad (h_{i1},\ldots,h_{in}),\\
\text{graph:}&\quad (V_i,E_i,X_i),\\
\text{hierarchy:}&\quad (H_i^{\mathrm{cell}},H_i^{\mathrm{region}}),\\
\text{distribution:}&\quad \widehat\mu_i,\\
\text{concept field:}&\quad M_i\in\mathbb R^{n_i\times K}.
\end{aligned}
```

Explanation resolution must match the slide object. A distribution model cannot
explain a particular spatial edge unless spatial geometry was retained outside
the distribution summary.

## 2. Resolution Map

```text
pixel       -> image attribution or generated morphology
patch       -> MIL attention, prototype, or deletion score
cell        -> cell graph attribution
region      -> coarsened graph or interpretable feature attribution
slide       -> risk, class, or multimodal representation
cohort      -> concept stability, bias, or calibration analysis
```

Interpolation between levels is a rendering or aggregation operation and should
not be mistaken for direct evidence at the finer level.

## 3. Same Map, Different Object

The same attention vector can be interpreted as patch evidence in a set model,
node routing in a graph model, or a time-conditioned selector in a survival
model. Its semantics come from the forward map, not from the plotted colors.

