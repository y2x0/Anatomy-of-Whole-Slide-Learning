# Attention Pooling

Attention pooling replaces a fixed average with a learned probability measure
over instances.

The core question:

```text
Which instances should define the slide statistic?
```

For instance states $u_{ij}$, attention pooling computes:

```math
a_{ij}
=
\frac{\exp(s_\theta(u_{ij}))}
{\sum_{\ell=1}^{n_i}\exp(s_\theta(u_{i\ell}))},
\qquad
z_i
=
\sum_{j=1}^{n_i}a_{ij}v_\theta(u_{ij}).
```

The surviving statistic is a learned weighted first moment:

```math
z_i
=
\mathbb{E}_{j\sim a_i}[v_\theta(u_{ij})].
```

## C/R/G/S Placement

```text
G:
    none in ABMIL unless geometry is already encoded in u_ij

C:
    score and value maps; DSMIL adds a critical-instance query

R:
    softmax-weighted first moment

S:
    slide-level loss shaping attention scores and values
```

Surviving statistic:

```math
z_i
=
\int v_\theta(u)\,d\nu_{i,\theta}(u),
\qquad
\nu_{i,\theta}
=
\sum_j a_{ij}\delta_{u_{ij}}.
```

## Files

- `01_attention_as_learned_measure.md`: ABMIL as a reweighted empirical measure.
- `02_gated_attention_temperature_and_gradients.md`: gated attention, softmax
  temperature, and gradient geometry.
- `03_dsmil_critical_instance_attention.md`: DSMIL-style max branch plus
  critical-instance attention.
- `04_failure_modes.md`: collapse, non-identifiability, and explanation traps.

## Anchor Papers

- Ilse, Tomczak, Welling. "Attention-based Deep Multiple Instance Learning."
  ICML 2018. https://arxiv.org/abs/1802.04712
- Li, Li, Eliceiri. "Dual-stream Multiple Instance Learning Network for Whole
  Slide Image Classification with Self-supervised Contrastive Learning."
  CVPR 2021. https://arxiv.org/abs/2011.08939
