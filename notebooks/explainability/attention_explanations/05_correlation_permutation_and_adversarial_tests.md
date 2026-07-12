# Correlation, Permutation, And Adversarial Tests

Primary anchor:

- Jain, Wallace. "Attention is not Explanation." NAACL 2019.
  https://arxiv.org/abs/1902.10186

## Correlation With Gradient Importance

Let attention scores be `alpha_j` and an independent importance estimate be
`g_j`, such as gradient magnitude. Rank correlation is:

```math
\rho_{\mathrm{rank}}
=
\mathrm{corr}
\left(
\mathrm{rank}(\alpha),
\mathrm{rank}(g)
\right).
```

Low correlation shows disagreement between two operational definitions. It
does not establish which one is ground truth.

## Permutation Test

For permutation `pi`:

```math
\alpha_j^{\pi}
=
\alpha_{\pi(j)}.
```

Holding values fixed:

```math
z^{\pi}
=
\sum_j
\alpha_{\pi(j)}v_j.
```

Prediction sensitivity can be summarized by:

```math
\Delta_{\pi}
=
\left|
F(z^{\pi})-F(z)
\right|.
```

Small `Delta_pi` for very different permutations indicates that displayed
weight placement is not necessary for the current prediction.

## Adversarial Attention

Search for:

```math
\alpha^{\mathrm{adv}}
=
\arg\max_{\alpha'\in\Delta^{n-1}}
D
\left(
\alpha',\alpha
\right)
-
\lambda
\left|
F(V\alpha')-F(V\alpha)
\right|.
```

The first term pushes explanations apart; the second preserves output.

## Jensen-Shannon Distance

For attention distributions `p,q`:

```math
\mathrm{JS}(p\|\|q)
=
\frac{1}{2}
\mathrm{KL}
\left(
p\|\|m
\right)
+
\frac{1}{2}
\mathrm{KL}
\left(
q\|\|m
\right),
```

where:

```math
m
=
\frac{p+q}{2}.
```

It is finite for distributions with nonmatching supports and was used to
quantify alternative attention divergence.

## Test Interpretation

Permutation and adversarial tests estimate necessity and uniqueness under a
fixed-value intervention. They do not test localization accuracy, causal tissue
importance, or whether retraining without learned attention harms performance.

## WSI Scaling

As bag size grows, the simplex dimension grows and the value matrix remains
rank-limited. The feasible set of alternative attention maps can expand,
making uniqueness especially demanding for gigapixel bags.
