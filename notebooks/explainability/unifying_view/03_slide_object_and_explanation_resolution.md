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

## Resolution As A Projection

Let E_i be an explanation defined on the native slide object and let Pi_l render
or aggregate it at resolution level l:

```math
e_i^{(\ell)}
=
\Pi_{\ell}\left(E_i;\mathcal O_i\right),
\qquad
\mathcal O_i\in
\{\text{set},\text{sequence},\text{graph},\text{hierarchy},\text{measure}\}.
```

The rendering map is generally many-to-one. Two patch-level explanations can
produce the same region heatmap, and two different graph edge explanations can
produce the same node score. Consequently,

```math
\Pi_{\ell}(E_i)=\Pi_{\ell}(E_i')
\centernot\Longrightarrow
E_i=E_i'.
```

The reverse error is also possible: interpolating a slide score onto patches
creates an array without proving that the model computed patch-level evidence.
An explanation report should name both the native object and the projection used
to display it.
