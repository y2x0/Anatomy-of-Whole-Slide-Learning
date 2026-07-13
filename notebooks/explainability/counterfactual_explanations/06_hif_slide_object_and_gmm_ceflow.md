# Human-Interpretable Slide Features and GMM-CeFlow

Benkirane et al. move counterfactual search from gigapixel images to a
high-dimensional vector of human-interpretable slide features.

## 1. Slide Object

Let

```math
u_i\in\mathbb R^p
```

contain tissue-region areas, cell- and tissue-type proportions, and cell
clustering properties extracted from a WSI. A coordinate edit can therefore be
described in pathology-facing terms.

This object is lossy:

```math
X_i\xrightarrow{\Phi_{\mathrm{HIF}}}u_i.
```

Different spatial arrangements and cellular morphologies can share the same
feature vector.

## 2. Normalizing Flow

An invertible RealNVP map sends feature vectors to latent variables:

```math
z=T_{\theta}(u),
\qquad
u=T_{\theta}^{-1}(z).
```

Class-conditional latent density is modeled by Gaussian components:

```math
p_Z(z\mid Y=c)
=\mathcal N(z;\mu_c,\Sigma_c).
```

By change of variables,

```math
\log p_U(u\mid c)
=\log p_Z(T_{\theta}(u)\mid c)
+\log\left|
\det J_{T_{\theta}}(u)
\right|.
```

The flow learns a tractable geometry in which class regions have quadratic
boundaries.

## 3. Sparse Class Geometry

The paper applies an L1 penalty to Gaussian means to encourage sparse class
separation. A schematic density-training objective with that penalty is:

```math
\mathcal L_{\mathrm{flow}}
=-\sum_i\log p_U(u_i\mid y_i)
+\lambda\sum_c\|\mu_c\|_1.
```

The displayed equation makes the role of the reported mean penalty explicit;
the short paper does not provide this full loss as a numbered equation. Sparse
means can reduce the number of latent directions carrying class separation.
They do not directly guarantee sparse changes after the nonlinear inverse flow.

## 4. Counterfactual Map

For original latent point

```math
z_{\mathrm{org}}=T_{\theta}(u),
```

GMM-CeFlow projects toward the boundary between original and target Gaussian
classes and decodes:

```math
u_{\mathrm{cf}}=T_{\theta}^{-1}(z_{\mathrm{cf}}).
```

The resulting statement concerns changes in extracted HIF coordinates under
the learned flow model, not direct pixel edits to the original slide.
