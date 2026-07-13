# Graph Readouts And Surviving Statistics

The graph context operator and the graph readout answer different questions:

```text
C:
    which node states can depend on which other nodes?

R:
    which statistic of the contextualized node field reaches the task head?
```

Let:

```math
\widetilde H_i
=
\left[
\widetilde h_{i1}^{\top};\ldots;\widetilde h_{in_i}^{\top}
\right]
\in
\mathbb{R}^{n_i\times d}.
```

## 1. Sum And Mean

Sum readout:

```math
z_i^{\mathrm{sum}}
=
\sum_{v=1}^{n_i}\widetilde h_{iv}.
```

Mean readout:

```math
z_i^{\mathrm{mean}}
=
\frac{1}{n_i}
\sum_{v=1}^{n_i}\widetilde h_{iv}.
```

For the empirical node measure:

```math
\widehat\mu_i
=
\frac{1}{n_i}
\sum_{v=1}^{n_i}\delta_{\widetilde h_{iv}},
```

the mean is its first moment:

```math
z_i^{\mathrm{mean}}
=
\int h\,d\widehat\mu_i(h).
```

The sum is the mean multiplied by node count:

```math
z_i^{\mathrm{sum}}
=
n_i z_i^{\mathrm{mean}}.
```

This distinction matters for HACT, where the cell-to-tissue sum can intentionally
preserve cellularity, and for variable-size WSI graphs, where sum can make node
count a shortcut.

## 2. Coordinatewise Max

```math
[z_i^{\mathrm{max}}]_r
=
\max_{1\le v\le n_i}
[\widetilde h_{iv}]_r.
```

Max preserves coordinatewise extrema but loses multiplicity. It is not a
single selected node in general because:

```math
v_r^{\star}
\in
\underset{v}{\arg\max}
[\widetilde h_{iv}]_r
```

can vary with `r`.

## 3. Global Attention

Let scores be:

```math
s_i
=
f_{\theta}(\widetilde H_i)
\in
\mathbb{R}^{n_i}.
```

Normalize over nodes:

```math
\alpha_i
=
\mathrm{softmax}(s_i),
```

and read out:

```math
z_i^{\mathrm{attn}}
=
\widetilde H_i^{\top}\alpha_i.
```

This is a weighted first moment. Its collision set is large: any two node
fields with the same score-weighted mean are identical to the next head.
Moreover, the weights are not explanations by themselves. A graph node may
receive a high readout weight because its contextualized state already contains
messages from many neighbors.

Patch-GCN uses global attention after coordinate graph context. Therefore its
attention score is a function of `h_tilde_iv`, not necessarily a function of
the original isolated patch.

## 4. Typed Readout

HEAT's pseudo-label pooling uses a fixed partition:

```math
V_{ia}
=
\left\{
v:\tau_i(v)=a
\right\}.
```

The readout is a tuple of type-conditioned statistics:

```math
Z_i^{\mathrm{typed}}
=
\left[
\mathcal{R}_{a}
\left(
\{\widetilde h_{iv}:v\in V_{ia}\}
\right)
\right]_{a\in\mathcal{T}}.
```

Compared with one global mean, this preserves a semantic coordinate system:

```text
the statistic for type a cannot be exchanged with the statistic for type b
without changing the representation.
```

It still loses within-type arrangements and any type information removed by
the upstream pseudo-labeling map.

## 5. Hierarchical Readout

HACT inserts an additional operator before the final readout:

```math
H_i^{\mathrm{cell}}
\xrightarrow{\;B_i^{\top}\;}
U_i^{\mathrm{tissue}}
\xrightarrow{\;\mathcal{C}_{\mathrm{tissue}}\;}
\widetilde H_i^{\mathrm{tissue}}
\xrightarrow{\;\sum_b\;}
z_i.
```

The surviving statistic is a sum of coarse nodes whose inputs are sums of fine
nodes. In a linearized model:

```math
z_i
=
\mathbf{1}^{\top}
\widetilde A_i^{\mathrm{tissue}}
\left[
H_i^{\mathrm{tissue}}\middle\Vert B_i^{\top}
\widetilde A_i^{\mathrm{cell}}H_i^{\mathrm{cell}}
\right].
```

This makes the hierarchy explicit in the statistic rather than treating it as
a visualization added after a flat MIL model.

## 6. Readout Is Task-Relative

Let a task head be:

```math
\widehat y_i
=
\mathcal{H}_{\theta}(z_i).
```

The same graph readout can feed different tasks:

```math
\eta_i
=
w_{\mathrm{cox}}^{\top}z_i,
```

```math
\widehat p_i
=
\mathrm{softmax}(W_{\mathrm{cls}}z_i+b),
```

```math
\widehat q_i(t_k)
=
\sigma\left(w_k^{\top}z_i+b_k\right).
```

The surviving statistic is judged by the head and objective. A statistic that
works for classification may be insufficient for calibrated horizon-specific
risk or retrieval.

## 7. Comparison

| Method | Contextual node field | Readout | Primary surviving statistic |
| --- | --- | --- | --- |
| Patch-GCN | coordinate-neighborhood, multiscale graph states | global attention | weighted first moment of contextual states |
| HEAT | typed edge-attribute transformer states | PL-Pool then graph readout | type-conditioned summaries |
| WiKG | learned directed top-`K` states with dual interaction | mean or max | mean or coordinatewise extrema of relational states |
| HACT | cell graph then tissue graph states | tissue sum or layerwise sum | hierarchical coarse-scale sum |
