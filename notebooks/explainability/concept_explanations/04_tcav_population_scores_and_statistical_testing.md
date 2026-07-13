# TCAV Population Scores and Statistical Testing

## 1. Class-Level Score

For examples `X_c` from class `c`, the TCAV score is

```math
\mathrm{TCAV}_{C,c,l}
=\frac{1}{|X_c|}
\sum_{x\in X_c}
\mathbf 1\{S_{C,c,l}(x)>0\}.
```

It estimates

```math
\theta_{C,c,l}
=\mathbb P
\left(S_{C,c,l}(X)>0\mid Y=c\right)
```

under the empirical evaluation distribution. Magnitudes are discarded. Two
concepts can have identical TCAV scores while one has derivatives a thousand
times larger.

## 2. Sampling Variance

Conditional on a fixed CAV and independent class samples, the Bernoulli
approximation gives

```math
\mathrm{Var}(\widehat\theta\mid v_C)
=\frac{\theta(1-\theta)}{|X_c|}.
```

But the CAV is estimated too. With random reference resamples `r`, write

```math
\widehat\theta_r
=\mathrm{TCAV}(v_{C,r}).
```

Total uncertainty contains evaluation-sample variation and CAV-estimation
variation:

```math
\mathrm{Var}(\widehat\theta)
=\mathbb E_v[\mathrm{Var}(\widehat\theta\mid v)]
+\mathrm{Var}_v(\mathbb E[\widehat\theta\mid v]).
```

Reporting only one trained CAV hides the second term.

## 3. Original Significance Logic

Kim et al. retrain CAVs against multiple random counterexample sets, producing
scores

```math
\widehat\theta_1,\ldots,\widehat\theta_R.
```

The paper applies a two-sided t-test against the sign-symmetry null

```math
H_0:\mathbb E[\widehat\theta_r]=\frac12
```

and uses a Bonferroni correction for its paired hypotheses. This tests whether
positive sensitivities occur at a rate distinguishable from one half across
the repeated CAV constructions. A valid test unit is the independently
resampled concept/reference construction, not each patch derivative treated as
independent after sharing one CAV.

Sauter et al.'s pathology implementation describes its null as equality of the
concept and random-counterexample TCAV score distributions and uses a
significance level of `0.01`. That is a related implementation-level test, but
it should not be silently substituted for the original paper's stated
one-half null.

## 4. Pathology Dependence

WSI patches from the same patient are correlated. If every patch is counted as
an independent class example, the effective sample size is exaggerated. For
patient `i`, first construct a patient-level statistic:

```math
T_i
=\frac1{n_i}\sum_{j=1}^{n_i}
\mathbf 1\{S_{C,c,l}(x_{ij})>0\}.
```

Then resample or test at the patient level. Slide-level or patient-level splits
must also be used when learning concept directions to avoid leakage.

## 5. Interpretation Boundary

A significant TCAV score supports reproducible directional association under a
particular concept construction. Statistical significance does not establish
semantic correctness, causal use, or clinical validity.
