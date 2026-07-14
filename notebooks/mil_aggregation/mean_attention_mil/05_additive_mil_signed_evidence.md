# Additive MIL and Signed Evidence

## 1. Scope and source

Additive MIL moves the patch-level predictor inside the final bag sum. The
result is a model whose score has an exact finite decomposition into signed
instance evidence.

The primary source is Javed et al., Additive MIL: Intrinsically Interpretable
Multiple Instance Learning:

https://arxiv.org/abs/2206.01794

The central claim is narrower than "the model is explainable":

> In score space, under independent patch evidence and an explicit additive
> readout, removing one patch changes the score by exactly that patch's term.

This gives a stronger accounting identity than an attention heatmap. It also
imposes a real inductive bias: cross-instance interactions are absent from the
final score unless they are introduced upstream or through an additional
interaction term.

## 2. Bag and patch evidence

Let slide i contain n_i patch inputs x_ij and patch representations h_ij:

```math
h_{ij}
=
m_\theta(x_{ij})
\in\mathbb R^d.
```

For class c, let the patch evidence function be

```math
e_{ij}^{(c)}
=
\phi_{\theta,c}(h_{ij})
\in\mathbb R.
```

The additive slide score is

```math
F_i^{(c)}
=
b_c
+
\sum_{j=1}^{n_i}e_{ij}^{(c)}.
```

The bias b_c is the empty-bag or baseline score. The complete prediction may
be a sigmoid or softmax applied after the additive score.

For binary classification:

```math
p_i
=
\sigma(F_i),
\qquad
\sigma(t)
=
\frac{1}{1+\exp(-t)}.
```

For multiclass classification:

```math
p_i^{(c)}
=
\frac{\exp(F_i^{(c)})}
{\sum_{r=1}^{C}\exp(F_i^{(r)})}.
```

The additive decomposition is exact for logits, not generally for probabilities.

## 3. Position of the nonlinearity

Conventional attention MIL has the form

```math
F_i^{\mathrm{attn}}
=
\psi_\theta\!\left(
\sum_j\alpha_{ij}h_{ij}
\right).
```

Additive MIL has the form

```math
F_i^{\mathrm{add}}
=
b+\sum_j\phi_\theta(h_{ij}).
```

The order of sum and nonlinear prediction changes the function class:

```math
\psi\!\left(\sum_j u_j\right)
\ne
\sum_j\psi(u_j)
\quad\text{in general}.
```

The additive architecture chooses the second ordering so that each term
remains a named score contribution.

If a context encoder is used before evidence:

```math
h_{ij}
=
\mathcal C_\theta(H_i)_j,
\qquad
F_i
=
b+\sum_j\phi_\theta(h_{ij}),
```

the final score is still additive in the contextualized representations, but
the evidence term can depend on the full bag through C. Exact deletion then
requires specifying whether context is recomputed.

## 4. Exact deletion identity

Let H_i minus j be the bag with patch j removed and all remaining patch
representations unchanged. Then

```math
F_i^{(c)}(H_i)
=
b_c+
e_{ij}^{(c)}
+
\sum_{\ell\ne j}e_{i\ell}^{(c)}.
```

The reduced score is

```math
F_i^{(c)}(H_i^{-j})
=
b_c+
\sum_{\ell\ne j}e_{i\ell}^{(c)}.
```

Therefore

```math
F_i^{(c)}(H_i)
-
F_i^{(c)}(H_i^{-j})
=
e_{ij}^{(c)}.
```

This is an exact finite intervention identity in score space. It does not
require a local approximation, a gradient, or an attention interpretation.

For a group G:

```math
F_i^{(c)}(H_i)
-
F_i^{(c)}(H_i^{-G})
=
\sum_{j\in G}e_{ij}^{(c)}.
```

The group credit is exactly additive when the remaining evidence terms are
unchanged by deletion.

## 5. Baseline and empty bag

The exact identity depends on the baseline convention. With an empty-bag
baseline:

```math
F_i^{(c)}(\varnothing)
=
b_c.
```

Then

```math
F_i^{(c)}(H_i)
-
F_i^{(c)}(\varnothing)
=
\sum_j e_{ij}^{(c)}.
```

The bias is not evidence from any patch. It is the score assigned to the
reference state.

If the implementation includes a nonzero reference bag H_i^0, define

```math
\widetilde e_{ij}^{(c)}
=
e_{ij}^{(c)}-e_{ij}^{(c,0)}.
```

The completeness relation becomes

```math
F_i^{(c)}(H_i)-F_i^{(c)}(H_i^0)
=
\sum_j\widetilde e_{ij}^{(c)}
```

only when the same patch indexing and additive decomposition are used in both
states.

## 6. Signed evidence

For a target class c:

```math
e_{ij}^{(c)}>0
\quad\Longrightarrow\quad
\text{patch j raises the class score relative to the baseline},
```

```math
e_{ij}^{(c)}<0
\quad\Longrightarrow\quad
\text{patch j lowers the class score relative to the baseline}.
```

The sign is meaningful in score or log-odds space. It is not the same as a
probability change:

```math
p_i-p_i^{-j}
=
\sigma(F_i)
-
\sigma(F_i-e_{ij}).
```

The sigmoid compresses large positive and negative score changes near its
extremes. A large signed score can therefore produce a small probability
difference when the slide is already very confident.

For a class contrast:

```math
e_{ij}^{(c,c')}
=
e_{ij}^{(c)}-e_{ij}^{(c')}.
```

Then

```math
F_i^{(c)}-F_i^{(c')}
=
(b_c-b_{c'})
+
\sum_j e_{ij}^{(c,c')}.
```

This is often the cleanest explanation for a multiclass margin.

## 7. Linear evidence subcase

If the patch evidence is linear:

```math
e_{ij}^{(c)}
=
\left(w_c\right)^{\mathsf T}h_{ij},
```

then

```math
F_i^{(c)}
=
b_c+
\left(w_c\right)^{\mathsf T}
\sum_jh_{ij}.
```

The model is a linear classifier on the sum representation. Its exact
per-patch credit is

```math
e_{ij}^{(c)}
=
\left(w_c\right)^{\mathsf T}h_{ij}.
```

This subcase exposes the connection to sum pooling. Mean pooling is recovered
by scaling the evidence by one over n_i:

```math
e_{ij}^{(c,\mathrm{mean})}
=
\frac{1}{n_i}
\left(w_c\right)^{\mathsf T}h_{ij}.
```

The difference is count sensitivity:

```math
\sum_j e_{ij}^{(c)}
\quad\text{versus}\quad
\frac{1}{n_i}\sum_j e_{ij}^{(c)}.
```

## 8. Count and burden

Unnormalized additivity preserves total mass. If every patch contributes a
positive average evidence mu_c:

```math
\mathbb E[F_i^{(c)}\mid n_i]
\approx
b_c+n_i\mu_c.
```

The score grows with patch count. This can represent disease burden or area,
but can also encode tissue amount, scanner resolution, or tiling density.

If patch density changes while physical tissue burden is fixed, sum evidence can
change:

```math
n_i\longmapsto rn_i
\quad\Longrightarrow\quad
F_i^{(c)}-b_c
\longmapsto
r\left(F_i^{(c)}-b_c\right)
\quad\text{under uniform replication}.
```

A normalized additive score can be

```math
F_i^{(c)}
=
b_c+
\frac{1}{n_i}\sum_j e_{ij}^{(c)}.
```

This restores replication invariance but loses absolute burden. The
normalization choice is part of the representation design.

## 9. Exact derivatives

For independent patch evidence, the derivative of the score is local:

```math
\frac{\partial F_i^{(c)}}{\partial h_{ij}}
=
\frac{\partial e_{ij}^{(c)}}{\partial h_{ij}}.
```

For distinct patches j and ell:

```math
\frac{\partial^2F_i^{(c)}}
\partial h_{ij}\partial h_{i\ell}^{\mathsf T}
=
0.
```

There are no cross-instance score interactions at the final additive operator.
The Hessian is block diagonal across independent patches:

```math
\nabla_{H_i}^2F_i^{(c)}
=
\mathrm{blockdiag}\!\left(
\nabla^2e_{i1}^{(c)},\ldots,
\nabla^2e_{in_i}^{(c)}
\right).
```

This zero cross-derivative is a structural statement, not an empirical
observation.

If the patch encoder is contextual:

```math
h_{ij}
=
\mathcal C_\theta(H_i)_j,
```

then cross-instance derivatives can reappear through C:

```math
\frac{\partial^2F_i}
\partial x_{ij}\partial x_{i\ell}^{\mathsf T}
\ne 0
\quad\text{even though the final evidence sum is additive in h}.
```

The phrase intrinsically additive must therefore specify the level at which
the independence assumption holds.

## 10. Shapley boundary

For a set function with coalition score

```math
F_i^{(c)}(S)
=
b_c+
\sum_{j\in S}e_{ij}^{(c)},
```

the marginal contribution of patch j to any coalition S not containing j is

```math
F_i^{(c)}(S\cup\{j\})
-
F_i^{(c)}(S)
=
e_{ij}^{(c)}.
```

The Shapley value averages this marginal contribution over coalitions:

```math
\varphi_{ij}^{(c)}
=
\sum_{S\subseteq N_i\setminus\{j\}}
\frac{|S|!(n_i-|S|-1)!}{n_i!}
\left[
F_i^{(c)}(S\cup\{j\})-F_i^{(c)}(S)
\right].
```

Because every bracket equals e_ij:

```math
\varphi_{ij}^{(c)}
=
e_{ij}^{(c)}.
```

This is the exact Shapley boundary for additive set functions. It fails when
the patch term depends on the coalition:

```math
e_{ij}^{(c)}(S)
\ne
e_{ij}^{(c)}(S')
```

or when pairwise interactions are present.

## 11. Pairwise interaction extension

An interaction-augmented model is

```math
F_i
=
b+
\sum_j e_{ij}
+
\sum_{j<\ell}r_{ij\ell}.
```

Deleting patch j changes

```math
F_i(H_i)-F_i(H_i^{-j})
=
e_{ij}
+
\sum_{\ell\ne j}r_{ij\ell}.
```

The single-patch score e_ij is no longer the full deletion credit. The
interaction remainder is

```math
\iota_{ij}
=
\sum_{\ell\ne j}r_{ij\ell}.
```

Additivity can be preserved while adding interactions through an expanded
explanation vocabulary, but the simple one-score-per-patch claim is lost.

For pairwise-only interaction, the mixed Hessian is

```math
\frac{\partial^2F_i}
\partial h_{ij}\partial h_{i\ell}^{\mathsf T}
=
\frac{\partial^2r_{ij\ell}}
\partial h_{ij}\partial h_{i\ell}^{\mathsf T}.
```

This gives a direct diagnostic for whether the architecture has cross-instance
score interactions.

## 12. Weak supervision

The observed target is a slide label:

```math
Y_i^{\mathrm{obs}}
=
y_i.
```

Patch evidence terms are latent model outputs:

```math
\{e_{ij}^{(c)}\}_{j=1}^{n_i}
\not\equiv
\{Y_{ij}^{\mathrm{true}}\}_{j=1}^{n_i}.
```

Training may minimize bag cross-entropy:

```math
\mathcal L_{\mathrm{bag}}
=
\sum_i
\mathrm{CE}\!\left(
\mathrm{softmax}(F_i),y_i
\right).
```

The exact score decomposition does not make patch signs ground-truth labels.
It only makes the model's score accounting auditable.

If the bag label is positive when at least one patch is positive, an additive
sum imposes a burden-like hypothesis:

```math
F_i
\approx
b+\sum_j e_{ij}.
```

This differs from an OR-like hypothesis:

```math
y_i
=
\mathbf 1\{\exists j:Y_{ij}=1\}.
```

An additive model can represent an OR through sharply positive evidence and a
threshold, but the relationship depends on calibration and background count.

## 13. Probability and logit explanations

For a binary score F and sigmoid output p:

```math
p
=
\sigma(F).
```

The exact score-space patch credit is e_j. The exact probability-space
deletion effect is

```math
\Delta_j^{\mathrm{prob}}
=
\sigma(F)-\sigma(F-e_j).
```

The sum of probability deletion effects is not the total probability change:

```math
\sum_j\Delta_j^{\mathrm{prob}}
\ne
\sigma(F)-\sigma(b)
\quad\text{in general}.
```

The logit or log-odds space is the additive explanation space. A report should
not call signed score evidence a probability contribution.

## 14. Spatial rendering

Let patch j cover physical region R_ij. The score map is piecewise constant:

```math
E_i^{(c)}(x)
=
\sum_j e_{ij}^{(c)}
\mathbf 1\{x\in R_{ij}\}.
```

If patches overlap, use an explicit rendering rule:

```math
E_i^{(c)}(x)
=
\frac{
\sum_{j:x\in R_{ij}}
w_{ij}(x)e_{ij}^{(c)}
}{
\sum_{j:x\in R_{ij}}w_{ij}(x)+\epsilon_0
}.
```

This projection preserves score additivity only at the patch level. Pixelwise
interpolation can redistribute visual mass and should not be used to claim a
new model-level decomposition.

## 15. Spatial interactions are absent at the readout

Two layouts with the same patch evidence multiset have the same score:

```math
\left\{
e_{i1},\ldots,e_{in_i}
\right\}
=
\left\{
e_{i1}',\ldots,e_{in_i}'
\right\}
\quad\Longrightarrow\quad
F_i=F_i'.
```

The additive readout cannot detect that two positive patches are adjacent,
separated, or arranged in a meaningful structure. Spatial context can be
encoded into h_ij before evidence computation, but then the evidence term may
carry contextual information that changes under deletion.

The geometry quotient is explicit:

```math
F_i(H_i,P_iC_i)
=
F_i(H_i,C_i)
\quad\text{if coordinates are not in the encoder}.
```

## 16. Calibration of evidence

The evidence terms can be shifted jointly with the bias:

```math
b'\!_c=b_c-\kappa,
\qquad
e_{ij}'^{(c)}=e_{ij}^{(c)}+\frac{\kappa}{n_i}
```

for a fixed bag size n_i, without changing the score. With variable bag sizes,
the same shift changes total evidence by kappa and is not globally
unidentifiable unless the parameterization accounts for count.

Evidence magnitudes are therefore calibrated relative to:

- the bias;
- the empty-bag convention;
- the number of patches;
- the normalization rule;
- the target score scale.

Comparing raw evidence across slides with different bag sizes requires a
declared normalization.

## 17. Failure modes

### 17.1 Missing interactions

The model cannot represent a label that depends only on a pair or spatial
configuration unless the patch encoder supplies that interaction.

### 17.2 Count shortcut

Unnormalized sums encode patch count and tissue area. This can be useful burden
signal or an acquisition artifact.

### 17.3 Burden mismatch

An OR-like diagnosis can be diluted or overcounted by an additive burden rule.

### 17.4 Evidence saturation

Large positive or negative logits can make probability effects appear small
even when score evidence is large.

### 17.5 Context ambiguity

If patch features are contextual, deleting one patch can alter other evidence
terms. The simple identity holds only when the remaining terms are held fixed.

### 17.6 Weak-label overinterpretation

Exact model credit is not ground-truth pathology localization. It proves what
the model score uses, not that the selected morphology is causal.

### 17.7 Rendering distortion

Interpolation and overlap averaging can make signed evidence appear smooth,
positive-only, or spatially continuous when the underlying patch terms are not.

## 18. Sanity checks

### Check A: completeness

Verify

```math
F_i^{(c)}-b_c
\stackrel{?}{=}
\sum_j e_{ij}^{(c)}.
```

This should hold up to numerical precision.

### Check B: deletion exactness

For random patch subsets G, compare:

```math
F_i^{(c)}(H_i)-F_i^{(c)}(H_i^{-G})
\stackrel{?}{=}
\sum_{j\in G}e_{ij}^{(c)}.
```

Failure indicates contextual recomputation, hidden normalization, or an
interaction term.

### Check C: probability distinction

Compare score evidence with probability deletion. The values should differ
except in a local small-logit regime.

### Check D: count sweep

Duplicate every patch uniformly, then duplicate only positive or background
patches. The result diagnoses whether the chosen normalization represents
burden, composition, or both.

### Check E: layout shuffle

Shuffle coordinates while preserving patch features. The score should remain
fixed when geometry is not encoded upstream.

### Check F: mixed-sign map

Report positive and negative evidence separately:

```math
E_i^{+}
=
\sum_{j:e_{ij}>0}e_{ij},
\qquad
E_i^{-}
=
\sum_{j:e_{ij}<0}e_{ij}.
```

Collapsing both into an absolute heatmap destroys the model's signed statement.

## 19. C/R/G/S placement

| Component | Additive MIL realization |
|---|---|
| Context operator | None at the final readout; optional context can be included in the patch encoder and must be declared. |
| Readout operator | Unnormalized or normalized sum of patch-level scalar evidence. |
| Geometry | No arrangement survives the additive sum unless encoded into patch evidence upstream. |
| Surviving statistic | Exact signed class score decomposition, with one evidence term per patch plus a baseline bias. |

The forward map is

```math
\text{patch}
\xrightarrow{\text{evidence function}}
\text{signed patch score}
\xrightarrow{\text{sum}}
\text{slide logit}
\xrightarrow{\text{sigmoid or softmax}}
\text{prediction}.
```

The explanation map is not post-hoc:

```math
\boxed{
\text{patch evidence}
=
\text{exact score-space deletion credit}
}
```

provided the intervention keeps the other evidence terms fixed.

## 20. Bottom line

Additive MIL earns a strong interpretability claim by placing the patch
predictor inside the final sum. The score decomposes into signed instance
evidence, and the Shapley value equals the patch term for the additive
coalition function.

That strength is also a limitation. The final score has no cross-instance
interaction unless contextual features or explicit interaction terms are
introduced. An exact model explanation is not a ground-truth pathology
explanation, and score additivity is not probability additivity.

The honest summary is:

```math
\boxed{
\text{additive MIL}
=
\text{signed patch evidence}
\;+\;
\text{exact score accounting}
\;-\;
\text{final-score interactions}
}
```

For a researcher who wants to know what each patch contributes to the model
score, this is a cleaner object than normalized attention. For a researcher
who needs spatial synergy or OR-style rare-positive logic, the additive
assumption must be tested rather than treated as universally correct.

## References

- Javed et al., "Additive MIL: Intrinsically Interpretable Multiple Instance
  Learning," https://arxiv.org/abs/2206.01794
- Ilse et al., "Attention-based Deep Multiple Instance Learning,"
  https://arxiv.org/abs/1802.04712
- Lu et al., "Data-efficient and weakly supervised computational pathology on
  whole-slide images," https://arxiv.org/abs/2004.09666
