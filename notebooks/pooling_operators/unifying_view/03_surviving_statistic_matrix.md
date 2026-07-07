# Surviving Statistic Matrix

Pooling defines an equivalence relation on slides. Two slides are equivalent
under a pooling operator when the operator maps them to the same slide-level
statistic.

If:

```math
z_i
=
\mathcal{R}(\widetilde H_i),
```

then the readout cannot distinguish slides $i$ and $i'$ whenever:

```math
\mathcal{R}(\widetilde H_i)
=
\mathcal{R}(\widetilde H_{i'}).
```

This note lists the statistic each family preserves and the failure mode that
follows from the induced equivalence class.

## Matrix

| Pooling family | Statistic preserved | Slides made equivalent | Main blind spot | Diagnostic toy example |
|---|---|---|---|---|
| Mean pooling | first moment | same average state | rare extremes, multimodality | one strong positive patch diluted by many negatives |
| Additive pooling | total burden | same summed evidence | confounds bag size with evidence | same fraction positive, different tissue area |
| Max pooling | largest score | same top instance | prevalence and context | one positive patch vs hundreds of positives |
| Top-k pooling | upper-tail average | same top-k multiset | choice of $k$ | disease area smaller or larger than $k$ |
| Quantile pooling | selected order statistic | same quantile | features outside chosen quantile | mixed distribution with same upper quantile |
| Generalized mean | temperature-dependent tail moment | same power mean | unstable interpolation between mean and max | high $p$ amplifies artifacts |
| Noisy-or pooling | product of negative probabilities | same non-event product | calibration and independence | many tiny false probabilities saturate slide risk |
| Attention pooling | weighted first moment | same attention-weighted average | attention collapse, non-identifiability | two attention maps yield same $z_i$ |
| Class-specific attention | class-conditioned weighted moment | same per-class readout | class competition and pseudo-label noise | high-attention negatives shaped as positives |
| Set Transformer PMA | seed-query summaries | same learned query outputs | seed redundancy, quadratic context cost | two seeds attend to same mode |
| Prototype pooling | component prevalence and residuals | same prototype histogram/residuals | prototype misspecification | two morphologies assigned to one component |
| Distribution pooling | chosen distribution functional | same histogram, MMD sketch, quantile, or OT summary | finite-sample noise and projection loss | same marginal distribution, different layout |
| Robust pooling | bounded-influence center | same robust center | removes rare diagnostic signal | rare tumor patches treated as outliers |
| Hierarchical pooling | composition of local statistics | same region summaries and global summary | premature compression | two regions have same mean but different patch mixtures |

## Equivalence Class View

Mean pooling creates the equivalence relation:

```math
H_i\sim_{\mathrm{mean}}H_{i'}
\quad\Longleftrightarrow\quad
\frac{1}{n_i}\sum_j h_{ij}
=
\frac{1}{n_{i'}}\sum_j h_{i'j}.
```

Max pooling creates:

```math
H_i\sim_{\max}H_{i'}
\quad\Longleftrightarrow\quad
\max_j g_\theta(h_{ij})
=
\max_j g_\theta(h_{i'j}).
```

Attention pooling creates:

```math
H_i\sim_{\mathrm{attn}}H_{i'}
\quad\Longleftrightarrow\quad
\sum_j a_\theta(h_{ij};H_i)v_\theta(h_{ij})
=
\sum_j a_\theta(h_{i'j};H_{i'})v_\theta(h_{i'j}).
```

Prototype pooling creates:

```math
H_i\sim_{\mathrm{proto}}H_{i'}
\quad\Longleftrightarrow\quad
T_{\mathrm{proto}}(\widehat\mu_i)
=
T_{\mathrm{proto}}(\widehat\mu_{i'}).
```

Hierarchical pooling creates:

```math
H_i\sim_{\mathrm{hier}}H_{i'}
\quad\Longleftrightarrow\quad
\mathcal{R}_{\mathrm{global}}
\left(
\{\mathcal{R}_{\mathrm{local}}(H_{im})\}_{m=1}^{M_i}
\right)
=
\mathcal{R}_{\mathrm{global}}
\left(
\{\mathcal{R}_{\mathrm{local}}(H_{i'm})\}_{m=1}^{M_{i'}}
\right).
```

The equivalence class is the real inductive bias. A pooling operator does not
only summarize information; it declares which differences between slides are
irrelevant.

## C/R/G/S Reading

For every family:

```text
C:
    changes the instance states before the statistic is computed

R:
    chooses the statistic that survives

G:
    determines whether spatial, graph, prototype, or scale structure can affect C or R

S:
    decides which differences are rewarded by the learning objective
```

The same $\mathcal{R}$ can behave very differently under different
$\mathcal{C}$. A mean after a graph neural network is a mean of
contextualized states. A mean before context is only a first moment of isolated
patch embeddings.

## Minimal Stress Tests

A useful pooling note should be able to answer these tests.

### Sparse Positive Test

Create two slides:

```math
H_A
=
\{p,n,n,\ldots,n\},
\qquad
H_B
=
\{n,n,\ldots,n\},
```

where $p$ is a strongly positive patch and $n$ is background. Mean pooling
fails when $n_i$ is large and $p$ has small mass. Max and noisy-or should
succeed if the patch scorer is calibrated.

### Prevalence Test

Create:

```math
H_A
=
\{p,n,n,\ldots,n\},
\qquad
H_B
=
\{p,p,p,\ldots,p,n,\ldots,n\}.
```

Max pooling cannot distinguish these if the strongest positive patch has the
same score. Mean, additive, top-k, and distribution pooling can distinguish
them if the representation separates $p$ and $n$.

### Layout Test

Create two slides with the same patch feature multiset but different
coordinates:

```math
\{(h_j,c_j)\}_{j=1}^{n}
\quad\mathrm{and}\quad
\{(h_j,c'_j)\}_{j=1}^{n}.
```

Any pooling operator that ignores $G$ must treat them as identical. Graph,
hierarchical, and geometry-aware distribution methods can separate them only if
layout affects either $\mathcal{C}$ or $\mathcal{R}$.

### Artifact Test

Create one slide with a true rare positive patch and one slide with a rare
artifact that receives a high score. Max pooling cannot distinguish them from
the score alone. Robust pooling may suppress both. Attention pooling can learn
to separate them only if the context, representation, or supervision provides
the relevant distinction.

## Design Rule

Before proposing a new pooling operator, specify:

```text
1. The statistic it preserves.
2. The slides it intentionally makes equivalent.
3. The slides it should separate.
4. The failure mode caused by that choice.
```

That is the mathematical design surface.
