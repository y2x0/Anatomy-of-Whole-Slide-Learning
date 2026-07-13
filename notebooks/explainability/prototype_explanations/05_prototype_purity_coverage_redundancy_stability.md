# Prototype Purity, Coverage, Redundancy, and Stability

Prototype explanations need population-level diagnostics in addition to
individual examples.

## 1. Purity

For the `m` nearest reference patches to prototype `k` and semantic annotation
`a_j`, define label purity

```math
\mathrm{Purity}_m(k)
=\max_a\frac1m\sum_{j\in N_m(p_k)}\mathbf 1\{a_j=a\}.
```

This measures neighborhood homogeneity under the available annotation, not
causal relevance. It is sensitive to `m`, reference-set composition, and
annotation granularity.

## 2. Coverage and Utilization

For threshold `t`, slide-level coverage and prototype utilization are

```math
\mathrm{Cov}_i(t)
=\frac1{n_i}\sum_j
\mathbf 1\left\{\max_k s_{ijk}\ge t\right\},
```

```math
\mathrm{Use}_k(t)
=\frac1N\sum_i\mathbf 1\{u_{ik}\ge t\}.
```

Low coverage indicates that much of the slide lies outside the prototype
dictionary. Near-zero utilization identifies dead prototypes; near-one
utilization can identify nonspecific prototypes.

## 3. Redundancy

A metric-space redundancy score is

```math
R_{k\ell}=\frac{p_k^\top p_\ell}{\|p_k\|_2\|p_\ell\|_2},
```

but functional redundancy is better measured by activation correlation

```math
R^{\mathrm{act}}_{k\ell}
=\mathrm{Corr}(u_{1k},\ldots,u_{Nk};
u_{1\ell},\ldots,u_{N\ell}).
```

Distinct latent vectors can be functionally redundant on the observed cohort.

## 4. Stability

Across two retrainings, prototype identities are permutation
ambiguous. Match sets by

```math
\sigma^\star
=\arg\min_{\sigma\in\mathfrak S_K}
\sum_{k=1}^K d(p_k^{(r)},p_{\sigma(k)}^{(r')}).
```

Then report matched distance, neighbor-set overlap, and score-credit agreement.
Stable predictions do not imply stable prototypes: several dictionaries can
span equivalent predictive functions.

## 5. Required Audit

A credible prototype explanation should report at least:

```text
purity + coverage + utilization + redundancy + seed stability + intervention
```

Nearest-neighbor mosaics alone test none of these jointly.
