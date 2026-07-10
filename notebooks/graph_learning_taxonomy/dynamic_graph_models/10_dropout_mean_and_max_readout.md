# Dropout, Mean Readout, And Max Readout

This note begins at the contextualized node field and derives dropout, the paper-level readout abstraction, mean pooling, and coordinatewise max pooling.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Contextualized Node Field

After knowledge-aware aggregation and dual-interaction fusion, one slide has:

```math
H^{+}
=
[h_1^{+};\ldots;h_n^{+}]
\in
\mathbb{R}^{n\times d}.
```

The released implementation applies dropout before readout. For dropout rate
`p`, write independent masks:

```math
m_{vr}
\sim
\mathrm{Bernoulli}(1-p),
```

and training-time node states:

```math
\overline h_{vr}
=
\frac{m_{vr}}{1-p}
h_{vr}^{+}.
```

Then:

```math
\mathbb{E}
\left[
\overline h_{vr}
\mid
h_{vr}^{+}
\right]
=
h_{vr}^{+}.
```

At evaluation time:

```math
\overline H
=
H^{+}.
```

The paper reports dropout rate:

```math
p
=
0.3.
```

## Paper-Level Readout

The paper writes the final prediction abstractly as:

```math
\widehat Y
=
\mathrm{Softmax}
\left(
\mathrm{Readout}(G)
\right),
```

and names mean or max pooling as examples of `Readout`.

This notation suppresses an important dimensional step. A graph readout
normally produces a feature vector in `R^d`, while a classifier must produce
`C` class logits. Unless `d=C`, a classification head is required:

```math
\mathbb{R}^{n\times d}
\xrightarrow{\mathcal{R}}
\mathbb{R}^{d}
\xrightarrow{\mathcal{H}}
\mathbb{R}^{C}
\xrightarrow{\mathrm{softmax}}
\Delta^{C-1}.
```

The released model contains this missing head explicitly.

## Mean Readout

Mean pooling gives:

```math
z^{\mathrm{mean}}
=
\frac{1}{n}
\sum_{v=1}^{n}
\overline h_v
\in
\mathbb{R}^{d}.
```

It preserves the first moment of the contextualized node distribution:

```math
\widehat\mu_H
=
\frac{1}{n}
\sum_{v=1}^{n}
\delta_{\overline h_v}.
```

Specifically:

```math
z^{\mathrm{mean}}
=
\int h
\,d\widehat\mu_H(h).
```

Distinct node fields with the same mean are indistinguishable after this
readout.

The released training script constructs WiKG with:

```text
pool='mean'
```

so mean pooling is the configured readout in that script.

## Max Readout

Coordinatewise max pooling gives:

```math
[z^{\mathrm{max}}]_r
=
\max_{1\le v\le n}
[\overline h_v]_r,
\qquad
r=1,\ldots,d.
```

The resulting vector need not equal any single node state. Different
coordinates can be supplied by different nodes:

```math
v_r^{\star}
\in
\underset{v}{\arg\max}
[\overline h_v]_r.
```

Max pooling preserves coordinatewise extrema but loses multiplicity. Repeating
the same maximizing morphology does not change the readout.
