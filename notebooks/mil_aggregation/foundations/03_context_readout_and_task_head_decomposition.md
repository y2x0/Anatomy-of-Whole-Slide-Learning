# Context, Readout, and Task-Head Decomposition

## 1. The Common Map

Let `H_i` be the instance matrix:

```math
H_i=[h_{i1}^{\top};\ldots;h_{in_i}^{\top}].
```

The unified MIL map is

```math
\widetilde H_i=\mathcal C(H_i;G_i,S_i),
\qquad
z_i=\mathcal R(\widetilde H_i),
\qquad
\widehat y_i=\mathcal H(z_i).
```

## 2. Operators

```text
C = context: none, attention interaction, transformer, graph, state-space;
R = readout: mean, max, attention, prototype, additive, distributional;
G = geometry: none, coordinates, order, graph, hierarchy;
S = supervision: bag label, pseudo-label, contrast, survival, multimodal cue.
```

The task head `H` determines which statistic is rewarded by the loss.

## 3. Factorization Ambiguity

An attention MIL model can be written as context-free weighted readout:

```math
z_i=\sum_j a_{ij}h_{ij},
```

or as a contextual map whose attention scores depend on all instances:

```math
a_i=\mathrm{softmax}(q(H_i)),
\qquad
z_i=\sum_j a_{ij}\widetilde h_{ij}.
```

These have different explanation semantics even when the final vector has the
same dimension.

## 4. Task Dependence

The same `z_i` can feed classification, Cox risk, hazard, retrieval, or
regression heads. A patch explanation is incomplete unless it names the head:

```math
\frac{\partial F_{\mathrm{class}}}{\partial h_{ij}}
\ne
\frac{\partial \eta_{\mathrm{Cox}}}{\partial h_{ij}}.
```

