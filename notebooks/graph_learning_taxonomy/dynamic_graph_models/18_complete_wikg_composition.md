# Complete WiKG Composition

This note closes the family with the complete operator composition, C/R/G/S factorization, asymptotic cost, and the two successive first-moment bottlenecks.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Complete Composition

The implementation-faithful map, with the triplet-score variant shown
explicitly, is:

```math
E
\xrightarrow{\mathrm{input\ projection}}
G
\xrightarrow{\mathrm{slide\ mean\ injection}}
\widetilde G
\xrightarrow{W_H,W_T}
(Q,T)
\xrightarrow{QT^{\top}/\sqrt d}
S
\xrightarrow{\text{rowwise top-}K}
(J,S_K)
\xrightarrow{\mathrm{softmax}}
\Omega
\xrightarrow{\mathrm{gather\ and\ interpolate}}
(T_K,R_K)
\xrightarrow{\tanh}
B_K
\xrightarrow{\mathrm{paper\ dot\ or\ code\ einsum}}
A_K
\xrightarrow{\mathrm{softmax}}
\Pi
\xrightarrow{\mathrm{weighted\ tail\ sum}}
M
\xrightarrow{\mathrm{dual\ interaction}}
H^{+}
\xrightarrow{\mathrm{dropout\ and\ readout}}
z
\xrightarrow{\mathrm{LayerNorm\ and\ linear}}
\ell
\xrightarrow{\mathrm{cross\ entropy}}
\mathcal{L}_{\mathrm{CE}}.
```

## C/R/G/S Factorization

Graph construction:

```math
\mathcal{G}_{\theta}(E)
=
(J,\Omega,R_K;Q,T).
```

Context:

```math
\mathcal{C}_{\theta}
\left(
Q,T,J,\Omega,R_K
\right)
=
H^{+}.
```

Readout:

```math
\mathcal{R}_{\theta}(H^{+})
=
z.
```

Head:

```math
\mathcal{H}_{\theta}(z)
=
\ell.
```

Supervision:

```math
\mathcal{S}
=
\left(
y,
\mathcal{L}_{\mathrm{CE}}
\right).
```

The full predictor is:

```math
\widehat p
=
\mathrm{softmax}
\circ
\mathcal{H}_{\theta}
\circ
\mathcal{R}_{\theta}
\circ
\mathcal{C}_{\theta}
\circ
\mathcal{G}_{\theta}
(E).
```

## Complexity Summary

Input and head-tail projections:

```math
\mathcal{O}(nd_0d+nd^2).
```

Dense compatibility:

```math
\mathcal{O}(n^2d)
```

time and:

```math
\mathcal{O}(n^2)
```

score memory.

Selected relation and context tensors:

```math
\mathcal{O}(nKd).
```

Readout:

```math
\mathcal{O}(nd)
```

for mean or max, with the attention gate adding its rowwise projection cost.

The all-pairs compatibility matrix is the asymptotic WSI-scale bottleneck when:

```math
n
\gg
K,d.
```

## Surviving Statistic

At each target, selected sources survive only through:

```math
m_v
=
\sum_{k=1}^{K}
\Pi[v,k]T_K[v,k,:].
```

At slide level under the released training script, contextualized targets
survive only through:

```math
z
=
\frac{1}{n}
\sum_{v=1}^{n}h_v^{+}.
```

WiKG therefore contains rich pair-dependent construction but two successive
first-moment bottlenecks: one over each target neighborhood and one over all
updated targets.
