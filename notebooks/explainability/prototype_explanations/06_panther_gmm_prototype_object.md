# PANTHER's GMM Prototype Object

PANTHER represents a slide by parameters of a slide-specific Gaussian mixture
whose component identities are shared across the cohort.

## 1. Generative Object

For slide `i`, assume patch embeddings are generated from

```math
p(h_{ij};\theta_i)
=\sum_{k=1}^K\pi_{ik}
\mathcal N(h_{ij};\mu_{ik},\Sigma_{ik}),
```

with

```math
\pi_{ik}\ge0,
\qquad
\sum_{k=1}^K\pi_{ik}=1.
```

The latent assignment variable takes one of `K` component indices and gives

```math
p(z_{ij}=k)=\pi_{ik},
\qquad
p(h_{ij}\mid z_{ij}=k)
=\mathcal N(h_{ij};\mu_{ik},\Sigma_{ik}).
```

The prototype is therefore not merely a single patch. Component `k` is a
distributional object described by prevalence, location, and dispersion.

## 2. Shared Identity, Slide-Specific State

Shared K-means centers `q_k` initialize component means, while each slide
estimates

```math
\theta_i=
\{(\pi_{ik},\mu_{ik},\Sigma_{ik})\}_{k=1}^K.
```

The shared initialization aligns component indices across slides. It does not
mathematically guarantee semantic identity after slide-specific estimation.
Component crossing or weakly occupied components can undermine the claim that
index `k` means the same morphology everywhere.

## 3. Representation

With diagonal covariance, PANTHER concatenates

```math
z_i=
\bigoplus_{k=1}^K
[\widehat\pi_{ik},
\widehat\mu_{ik}^{\top},
\mathrm{diag}(\widehat\Sigma_{ik})^{\top}]^{\top}
\in\mathbb R^{K(1+2d)}.
```

This retains an estimated distribution summary rather than a first moment.
Two slides with the same component means but different prevalence or variance
remain distinguishable.

## 4. Assumption Surface

The representation inherits:

```math
\text{finite mixture}+
\text{Gaussian components}+
\text{diagonal covariance}+
\text{shared component indexing}.
```

Interpretability arises from this factorization, not from the word prototype.
If a true tissue state is multimodal, one component may split it; if several
states overlap in embedding space, one component may merge them.
