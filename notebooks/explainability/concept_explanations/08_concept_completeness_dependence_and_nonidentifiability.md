# Concept Completeness, Dependence, and Nonidentifiability

## 1. Completeness as Predictive Sufficiency

Let concept map

```math
c(x)=(c_1(x),\ldots,c_K(x))
```

be evaluated against a fixed model output `F(x)`. A concept family is exactly
functionally complete for `F` on domain `X` if there exists a decoder `g` such
that

```math
F(x)=g(c(x))
\qquad
\text{for every }x\in\mathcal X.
```

Approximate completeness under distribution `P` can be measured by

```math
\inf_{g\in\mathcal G}
\mathbb E_{X\sim P}
[\ell(F(X),g(c(X)))].
```

This depends on decoder capacity. A powerful decoder can reconstruct outputs
from concepts that remain opaque to humans.

## 2. Completeness Is Not Correctness

A nuisance concept can be complete for a biased classifier. Conversely, a
clinically correct concept set can be incomplete if the model relies on
unrecognized texture. Therefore:

```math
\text{functional completeness}
\not\Rightarrow
\text{semantic validity}
\not\Rightarrow
\text{causal validity}.
```

## 3. Dependence and Double Counting

For concept vector `C`, define covariance

```math
\Sigma_C=\mathrm{Cov}(C).
```

If `Sigma_C` is nearly singular, multiple concepts encode similar variation.
Marginal sensitivity to each coordinate can double-count a shared latent
factor. Orthogonalization changes the coordinates and therefore changes their
semantic meaning; it is not a neutral correction.

## 4. Nonidentifiability

Suppose a linear concept decoder gives

```math
F(x)=w^{\top}c(x).
```

For any invertible matrix `A`, define

```math
c'(x)=Ac(x),
\qquad
w'=A^{-\top}w.
```

Then

```math
w'^{\top}c'(x)=w^{\top}c(x).
```

Prediction alone cannot identify the concept basis. Human labels, sparsity,
nonnegativity, prototype grounding, or cross-modal priors constrain the basis,
but each introduces its own assumptions.

## 5. Missing-Concept Test

Fit a residual predictor from the original representation after conditioning on
concepts:

```math
r(x)=F(x)-g(c(x)).
```

If a model can predict `r(x)` from the residual representation with substantial
held-out performance, the concept inventory omits functionally relevant
information. The result locates incompleteness; it does not name the missing
concept.

