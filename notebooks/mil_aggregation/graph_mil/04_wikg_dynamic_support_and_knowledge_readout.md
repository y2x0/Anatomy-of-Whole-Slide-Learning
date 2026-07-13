# WiKG Dynamic Support And Knowledge Readout

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
```

## 1. Slide Mean Injection

Let projected patch features be:

```math
G_i
=
\begin{bmatrix}
g_{i1}^{\top}\\
\vdots\\
g_{in_i}^{\top}
\end{bmatrix}
\in
\mathbb{R}^{n_i\times d}.
```

The released implementation injects the slide mean before graph construction:

```math
\overline g_i
=
\frac{1}{n_i}
\sum_{v=1}^{n_i}g_{iv},
```

```math
\widetilde g_{iv}
=
\frac{1}{2}
\left(
g_{iv}+\overline g_i
\right).
```

This is implementation-specific and should be separated from the paper's
displayed methodology. It means the learned topology can depend on a global
first moment before local graph context begins.

## 2. Learned Directed Topology

WiKG uses separate head and tail projections:

```math
q_{iv}
=
\widetilde g_{iv}W_H+b_H,
\qquad
t_{iu}
=
\widetilde g_{iu}W_T+b_T.
```

The directed score is:

```math
s_{ivu}
=
\frac{q_{iv}t_{iu}^{\top}}{\sqrt d}.
```

For target `v`, select the `K` largest source scores:

```math
\mathcal{N}_{i,K}(v)
=
\mathrm{TopK}_{u}\left(s_{ivu}\right).
```

The message-flow adjacency is:

```math
A_i[v,u]
=
\mathbf{1}\left\{u\in\mathcal{N}_{i,K}(v)\right\}.
```

The relation is generally directed because:

```math
s_{ivu}\neq s_{iuv}.
```

The support is dynamic in the input and parameters, but the released block does
not necessarily rebuild it after every graph layer. That is different from a
layerwise dynamic kNN GNN.

## 3. Knowledge-Aware Attention

On the selected support, construct a relation state using the first score
weights. The implementation-faithful two-stage abstraction is:

```math
\omega_{ivu}
=
\frac{\exp(s_{ivu})}
{\sum_{r\in\mathcal{N}_{i,K}(v)}\exp(s_{ivr})},
```

```math
r_{ivu}
=
\omega_{ivu}t_{iu}.
```

The relation state changes the second compatibility score. One paper-level
form is:

```math
a_{ivu}
=
t_{iu}
\left[
\tanh\left(q_{iv}+r_{ivu}\right)
\right]^{\top}.
```

Then the knowledge-aware distribution is:

```math
\pi_{ivu}
=
\frac{\exp(a_{ivu})}
{\sum_{r\in\mathcal{N}_{i,K}(v)}\exp(a_{ivr})}.
```

The neighbor message is a convex combination:

```math
m_{iv}
=
\sum_{u\in\mathcal{N}_{i,K}(v)}
\pi_{ivu}t_{iu}.
```

Thus the local graph statistic is a weighted first moment over a learned
directed neighborhood, with the weights themselves conditioned on a relation
state.

## 4. Dual Interaction

The updated node state has additive and coordinatewise multiplicative channels:

```math
h_{iv}^{+}
=
\phi_1
\left(
\left(q_{iv}+m_{iv}\right)W_1+b_1
\right)
+
\phi_2
\left(
\left(q_{iv}\odot m_{iv}\right)W_2+b_2
\right).
```

The product term is a second-order interaction in the learned feature basis:

```math
\left[q_{iv}\odot m_{iv}\right]_r
=
[q_{iv}]_r[m_{iv}]_r.
```

It is not automatically a biological interaction. It is a learned coordinate
interaction after the graph has compressed neighbors into `m_iv`.

## 5. Graph Readout

The released training configuration uses mean pooling:

```math
z_i^{\mathrm{mean}}
=
\frac{1}{n_i}
\sum_{v=1}^{n_i}h_{iv}^{+}.
```

The paper also discusses mean or max graph readout. Coordinatewise max is:

```math
[z_i^{\mathrm{max}}]_r
=
\max_{1\le v\le n_i}[h_{iv}^{+}]_r.
```

The classifier is:

```math
\widehat p_i
=
\mathrm{softmax}\left(z_iW_{\mathrm{cls}}+b_{\mathrm{cls}}\right).
```

## 6. C/R/G/S Placement

```text
C:
    hard top-k support, two-stage knowledge attention, and dual interaction

R:
    mean or max over updated head states; released training uses mean

G:
    learned directed feature geometry, optionally preceded by slide mean
    injection

S:
    slide classification cross-entropy shapes both relation projections and
    graph readout through the final loss
```

The critical surviving statistic is not “the top-k patches.” It is the field of
nodewise messages:

```math
\left\{
\sum_{u\in\mathcal{N}_{i,K}(v)}
\pi_{ivu}t_{iu}
\right\}_{v=1}^{n_i},
```

which is then compressed again by mean or max readout.
