# HIPPO Necessity, Sufficiency, And Greedy Search

Primary anchor:

- Kaczmarzyk et al. "Explainable AI for Computational Pathology Identifies
  Model Limitations and Tissue Biomarkers." 2024.
  https://arxiv.org/abs/2409.03080

## Necessity

For annotated tumor set `T`:

```math
\mathrm{Nec}(T)
=
F(P)
-
F(P\setminus T).
```

Large effect means tumor tissue is necessary for the model's current score
under this removal intervention.

## Retention Sufficiency

```math
\mathrm{Suff}_{\mathrm{retain}}(T)
=
F(T).
```

This tests whether tumor-only tissue retains a positive prediction after all
non-tumor patches are removed.

## Addition Sufficiency

For negative recipient bag `N`:

```math
\mathrm{Suff}_{\mathrm{add}}(T;N)
=
F(N\cup T)
-
F(N).
```

This tests tumor evidence in a new context and is stronger than retention in
its original specimen.

## Greedy Positive-Effect Search

At iteration `t`, current bag is `P_t`. Score each remaining patch by:

```math
s_i^{(t)}
=
F(P_t)
-
F
\left(
P_t\setminus\{p_i\}
\right).
```

Select:

```math
i_t^{\star}
=
\arg\max_i
s_i^{(t)},
```

then update:

```math
P_{t+1}
=
P_t
\setminus
\left\{
p_{i_t^{\star}}
\right\}.
```

Recomputing scores after each deletion captures some context dependence.

## Greedy Is Not Globally Optimal

The procedure optimizes one-step marginal effect. For target subset size `k`,
it need not solve:

```math
\arg\max_{|S|=k}
\left[
F(P)-F(P\setminus S)
\right].
```

Without submodularity or related structure, greedy approximation guarantees do
not hold.

## Negative-Effect Search

Selecting most negative one-removed effect identifies patches whose deletion
increases the target. These are inhibitory under the current bag context.

## Search Cost

Naive `k`-step greedy search requires:

```math
\sum_{t=0}^{k-1}
\left(
n-t
\right)
=
kn
-
\frac{k(k-1)}{2}
```

forward evaluations, in addition to original scores.

## Repeated Tissue

Greedy search can remove one representative from a redundant morphology and
observe little effect until redundancy is exhausted. Early rankings can
understate a collectively necessary tissue phenotype.
