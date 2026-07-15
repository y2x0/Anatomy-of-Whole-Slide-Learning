# Objective Geometry And Evaluation Matrix

Primary pathology sources used in this note:

- [HIPT](https://arxiv.org/abs/2206.02647)
- [RetCCL](https://pubmed.ncbi.nlm.nih.gov/36270093/)
- [UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/)
- [Virchow](https://arxiv.org/abs/2309.07778)
- [PLIP](https://pubmed.ncbi.nlm.nih.gov/37592105/)
- [CONCH](https://arxiv.org/abs/2307.12914)
- [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w)
- [TITAN](https://www.nature.com/articles/s41591-025-03982-3)
- [KEEP](https://arxiv.org/html/2412.13126v1)
- [Computational pathology foundation-model representational similarity analysis](https://arxiv.org/abs/2509.15482)

This note asks a narrower question than whether a pathology foundation model is
good:

```text
Which mathematical property of a representation is identified by a particular
evaluation protocol?
```

Linear probing, nearest-neighbor retrieval, zero-shot text classification,
MIL, segmentation, survival ranking, and calibration do not inspect the same
object. They are invariant to different transformations and can therefore rank
the same encoders differently without logical contradiction.

The paper-specific transfer-limits note asks what information a downstream
hypothesis class can recover. This note begins after that choice and asks which
changes of coordinates, score scale, memory, and sampling unit leave the
reported evaluation unchanged. Its object is the measurement functional, not a
second catalog of adaptation methods.

## 1. Evaluation Is A Functional, Not A Property Name

Let a frozen encoder be:

```math
f_{\phi}
:
\mathcal X
\longrightarrow
\mathcal Z.
```

Let the evaluation dataset be:

```math
\mathcal D
=
\left\{
    (X_i,Y_i,U_i)
\right\}_{i=1}^{N},
```

where `U_i` collects patient, institution, stain, scanner, and acquisition
metadata. An evaluation protocol is the tuple:

```math
\mathfrak E
=
\left(
    \mathcal D,
    \Pi,
    \mathcal H,
    \mathcal A,
    \Lambda,
    M,
    \mathcal B
\right).
```

Its components are:

```math
\begin{aligned}
\Pi
&:
\text{split and sampling rule},\\
\mathcal H
&:
\text{allowed downstream hypothesis class},\\
\mathcal A
&:
\text{fitting algorithm},\\
\Lambda
&:
\text{hyperparameter search space and selection rule},\\
M
&:
\text{reported metric},\\
\mathcal B
&:
\text{bootstrap or uncertainty procedure}.
\end{aligned}
```

The reported score is therefore:

```math
\widehat V
\left(
    f_{\phi};
    \mathfrak E
\right)
=
M
\left(
    \mathcal D_{\mathrm{test}},
    \widehat h_{\phi,\mathfrak E}
\right),
```

with fitted readout produced by the selected training procedure:

```math
\widehat h_{\phi,\mathfrak E}
=
\mathcal A_{\widehat\lambda}
\left(
    \mathcal D_{\mathrm{train}},
    f_{\phi}
\right),
```

after model selection under `\Lambda` and `\Pi`. It belongs to the empirical
argmin set only when the implemented procedure performs exact empirical risk
minimization over the selected class.

Thus a benchmark score is not a scalar attribute stored inside the encoder. It
is a functional of the encoder, dataset, split, readout class, optimizer,
selection rule, and metric.

## 2. Three Geometries Are Commonly Confused

The encoder induces a representation space:

```math
\mathcal Z_{\phi}
=
f_{\phi}(\mathcal X).
```

The readout defines a hypothesis geometry:

```math
\mathcal H\circ f_{\phi}
=
\left\{
    h\circ f_{\phi}
    :
    h\in\mathcal H
\right\}.
```

The metric defines a prediction comparison functional:

```math
\Gamma_M
\left(
    \widehat P,
    P^{\star}
\right).
```

The symbol `\Gamma_M` is not assumed to be a mathematical distance. Accuracy,
AUROC, concordance, and calibration summaries need not satisfy distance axioms.

These are different objects:

```math
\boxed{
\text{representation geometry}
\neq
\text{readout geometry}
\neq
\text{metric geometry}.
}
```

For example, an invertible anisotropic transformation can preserve every
unregularized linear decision boundary while changing cosine neighborhoods.
A strictly increasing score transformation can preserve every ranking metric
while changing every probability-calibration metric.

## 3. Population Value And Estimated Value

For loss `\ell` and readout class `\mathcal H`, define population transfer risk:

```math
\mathcal R_{\mathcal H}(f_{\phi})
=
\inf_{h\in\mathcal H}
\mathbb E
\left[
    \ell
    \left(
        h(f_{\phi}(X)),
        Y
    \right)
\right].
```

To obtain an exact decomposition, define target Bayes risk:

```math
\mathcal R_Q^{\star}
=
\inf_{g}
\mathbb E_Q
\left[
    \ell(g(X),Y)
\right],
```

where the infimum is over all measurable predictors for which the risk exists.
For candidate classes indexed by `\Lambda`, define source- and target-population
best readers:

```math
h_{P,\Lambda}^{\star}
\in
\arg\min_{
    \substack{
        \lambda\in\Lambda\\
        h\in\mathcal H_{\lambda}
    }
}
\mathcal R_P(h\circ f_{\phi}),
```

and:

```math
h_{Q,\Lambda}^{\star}
\in
\arg\min_{
    \substack{
        \lambda\in\Lambda\\
        h\in\mathcal H_{\lambda}
    }
}
\mathcal R_Q(h\circ f_{\phi}).
```

For the validation-selected candidate `\widehat\lambda`, let:

```math
\widetilde h_{\widehat\lambda}
\in
\arg\min_{h\in\mathcal H_{\widehat\lambda}}
\widehat{\mathcal R}_{\mathrm{train}}
\left(
    h\circ f_{\phi}
\right)
```

be an exact empirical minimizer, and let:

```math
\widehat h
=
\mathcal A_{\widehat\lambda}
\left(
    \mathcal D_{\mathrm{train}},
    f_{\phi}
\right)
```

be the implemented optimizer output. Assume the displayed minimizers exist;
otherwise replace each minimum by its infimum and use minimizing sequences.
The reported test risk then obeys the telescoping identity:

```math
\begin{aligned}
&\widehat{\mathcal R}_{\mathrm{test}}
\left(
    \widehat h\circ f_{\phi}
\right)
-
\mathcal R_Q^{\star}\\
&=
\underbrace{
\widehat{\mathcal R}_{\mathrm{test}}
\left(
    \widehat h\circ f_{\phi}
\right)
-
\mathcal R_Q
\left(
    \widehat h\circ f_{\phi}
\right)
}_{\text{test estimation}}\\
&\quad+
\underbrace{
\mathcal R_Q
\left(
    \widehat h\circ f_{\phi}
\right)
-
\mathcal R_Q
\left(
    \widetilde h_{\widehat\lambda}
    \circ f_{\phi}
\right)
}_{\text{optimization}}\\
&\quad+
\underbrace{
\mathcal R_Q
\left(
    \widetilde h_{\widehat\lambda}
    \circ f_{\phi}
\right)
-
\mathcal R_Q
\left(
    h_{P,\Lambda}^{\star}
    \circ f_{\phi}
\right)
}_{\text{finite-sample fitting and selection}}\\
&\quad+
\underbrace{
\mathcal R_Q
\left(
    h_{P,\Lambda}^{\star}
    \circ f_{\phi}
\right)
-
\mathcal R_Q
\left(
    h_{Q,\Lambda}^{\star}
    \circ f_{\phi}
\right)
}_{\text{source-to-target transport}}\\
&\quad+
\underbrace{
\mathcal R_Q
\left(
    h_{Q,\Lambda}^{\star}
    \circ f_{\phi}
\right)
-
\mathcal R_Q^{\star}
}_{\text{representation and candidate-class approximation}}.
\end{aligned}
```

The first three terms may have either sign. The source-to-target transport term
is nonnegative because `h_{Q,\Lambda}^{\star}` minimizes target risk over the
same candidate family, and the final approximation term is nonnegative because
Bayes risk minimizes over all measurable predictors. No independence is
assumed. The value of the decomposition is that each term is defined by
explicit intermediate objects rather than by an unnamed residual.

If the same test set chooses hyperparameters and reports final performance,
then:

```math
\mathbb E
\left[
    \max_{\lambda\in\Lambda}
    \widehat V_{\mathrm{test}}(\lambda)
\right]
\geq
\max_{\lambda\in\Lambda}
\mathbb E
\left[
    \widehat V_{\mathrm{test}}(\lambda)
\right].
```

For a finite candidate set, or a measurable integrable family for which the
maximum is well defined, the inequality follows from convexity of the maximum.
It formalizes winner's curse: more candidate settings can increase the selected
test score even when their true values are unchanged.

## 4. Evaluation Equivalence Classes

Two encoders are equivalent under evaluation protocol `\mathfrak E` when they
have the same population value:

```math
f_1
\sim_{\mathfrak E}
f_2
\quad\Longleftrightarrow\quad
V(f_1;\mathfrak E)
=
V(f_2;\mathfrak E).
```

A stronger behavioral equivalence requires the same induced prediction family:

```math
\mathcal H\circ f_1
=
\mathcal H\circ f_2.
```

For a transformation group `\mathcal G`, an evaluation is invariant when:

```math
V(g\circ f;\mathfrak E)
=
V(f;\mathfrak E)
\qquad
\forall g\in\mathcal G.
```

The full score-equivalence class is:

```math
[f]_{\mathfrak E}
=
\left\{
    f'
    :
    V(f';\mathfrak E)
    =
    V(f;\mathfrak E)
\right\},
```

while the orbit generated by a known invariance group is:

```math
\mathrm{Orb}_{\mathcal G_{\mathfrak E}}(f)
=
\left\{
    g\circ f
    :
    g\in\mathcal G_{\mathfrak E}
\right\}.
```

In general:

```math
\mathrm{Orb}_{\mathcal G_{\mathfrak E}}(f)
\subseteq
[f]_{\mathfrak E}.
```

Equal scalar scores can occur for encoders unrelated by the identified
transformation group. In either case, the evaluation does not identify a
unique coordinate system.

## 5. Unregularized Linear-Probe Invariance

Let the frozen representation be:

```math
z
=
f_{\phi}(X)
\in
\mathbb R^d.
```

A multiclass affine probe has logits:

```math
g_{W,b}(z)
=
Wz+b,
```

where:

```math
W\in\mathbb R^{C\times d},
\qquad
b\in\mathbb R^C.
```

Consider an invertible affine change of coordinates:

```math
z'
=
Az+a,
\qquad
A\in\mathrm{GL}(d).
```

Choose:

```math
W'
=
WA^{-1},
\qquad
b'
=
b-WA^{-1}a.
```

Then:

```math
\begin{aligned}
g_{W',b'}(z')
&=
WA^{-1}(Az+a)
+
b-WA^{-1}a\\
&=
Wz+b.
\end{aligned}
```

Therefore:

```math
\boxed{
\mathcal H_{\mathrm{aff}}
\circ
f
=
\mathcal H_{\mathrm{aff}}
\circ
(A f+a)
}
```

for every invertible `A`.

Population unregularized affine-probe risk is invariant to the full affine
group:

```math
\mathcal G_{\mathrm{linear}}
=
\mathrm{GL}(d)
\ltimes
\mathbb R^d.
```

A linear probe therefore cannot identify angles, coordinate scales, or cosine
neighborhoods. It identifies whether the target is accessible through an
affine decision map.

## 6. Regularization Breaks General Affine Invariance

Suppose the fitted probe has an unpenalized bias, fixed positive regularization
strength, and minimizes:

```math
\widehat L(W,b)
+
\lambda
\lVert W\rVert_F^2.
```

Here:

```math
\lambda>0.
```

Under the affine reparameterization above:

```math
\lVert W'\rVert_F^2
=
\left\lVert
    WA^{-1}
\right\rVert_F^2.
```

This equals the original penalty for all `W` only when:

```math
A^{-1}A^{-\mathsf T}
=
I,
```

which is the orthogonal case:

```math
A\in O(d).
```

The translation term is still absorbed by the unpenalized bias. Consequently,
the practical invariance group is the Euclidean group:

```math
\boxed{
\mathcal G_{\mathrm{ridge\ probe}}
=
O(d)
\ltimes
\mathbb R^d,
}
```

not the full affine group. If the bias is absent or penalized, translation
invariance can also fail.

Feature standardization, optimization tolerances, early stopping, and the
regularization search grid further modify the effective invariance group. Two
papers both reporting a linear probe need not be evaluating the same
hypothesis class in practice.

## 7. A Linear-Separability Counterexample

Let four embeddings be:

```math
z_1=(-1,-1),
\quad
z_2=(-1,1),
\quad
z_3=(1,-1),
\quad
z_4=(1,1),
```

with class label determined by the sign of the first coordinate:

```math
y(z)
=
\mathbb 1\{z_1>0\}.
```

Apply anisotropic scaling:

```math
A_{\epsilon}
=
\begin{bmatrix}
\epsilon & 0\\
0 & \epsilon^{-1}
\end{bmatrix},
\qquad
\epsilon>0.
```

The transformed points remain perfectly linearly separable for every positive
`\epsilon`. However, their angular relations converge toward the second axis
as `\epsilon` approaches zero. Thus linear-probe accuracy remains one while
cosine retrieval can become dominated by the task-irrelevant second
coordinate.

This is not a training failure. It is an evaluation-group mismatch.

## 8. Euclidean Retrieval Geometry

Given memory embeddings:

```math
\mathcal M
=
\left\{
    (z_j,y_j)
\right\}_{j=1}^{m},
```

Euclidean retrieval returns:

```math
\mathcal N_K(z)
=
\arg\min_{
    \substack{
        J\subseteq\{1,\ldots,m\}\\
        |J|=K
    }
}
\sum_{j\in J}
\lVert z-z_j\rVert_2^2.
```

Under:

```math
z'
=
cQz+a,
\qquad
c>0,
\qquad
Q^{\mathsf T}Q=I,
```

distances transform as:

```math
\lVert z'-z_j'\rVert_2^2
=
c^2
\lVert z-z_j\rVert_2^2.
```

Every distance ranking is preserved. Hence Euclidean nearest-neighbor
retrieval is invariant to translation, orthogonal transformation, and common
positive scaling.

For a general invertible matrix `A`:

```math
\lVert Az-Az_j\rVert_2^2
=
(z-z_j)^{\mathsf T}
A^{\mathsf T}A
(z-z_j).
```

The transformation changes Euclidean retrieval into Mahalanobis retrieval with
metric tensor:

```math
M_A
=
A^{\mathsf T}A.
```

Therefore general affine transformations preserve linear readout families but
not Euclidean neighborhoods.

## 9. Cosine Retrieval Geometry

For nonzero vectors, cosine similarity is:

```math
s_{\cos}(z,z_j)
=
\frac{z^{\mathsf T}z_j}
{\lVert z\rVert_2\lVert z_j\rVert_2}.
```

It is invariant to independent positive radial rescaling:

```math
s_{\cos}
\left(
    c(z)z,
    c(z_j)z_j
\right)
=
s_{\cos}(z,z_j),
```

and to a common orthogonal transformation:

```math
s_{\cos}(Qz,Qz_j)
=
s_{\cos}(z,z_j).
```

It is not translation invariant:

```math
s_{\cos}(z+a,z_j+a)
\neq
s_{\cos}(z,z_j)
```

in general. Nor is it invariant to anisotropic scaling.

The cosine-retrieval invariance family therefore differs from both affine
linear probing and Euclidean retrieval:

```math
\mathcal I_{\cos}
=
\left\{
    z\mapsto c(z)Qz
    :
    c(z)>0,
    Q\in O(d)
\right\}.
```

This family statement assumes that the same transformation rule is applied to
queries and memory items and that ties are handled consistently. It is not
called a group because an arbitrary input-dependent radial rescaling need not
be invertible. Restricting to bijective ray-preserving maps yields a genuine
transformation group.

## 10. Prototype Geometry

For class `c`, a Euclidean prototype is:

```math
\widehat\mu_c
=
\frac{1}{n_c}
\sum_{i:Y_i=c}
z_i.
```

Nearest-prototype classification uses:

```math
\widehat y(z)
=
\arg\min_c
\lVert z-\widehat\mu_c\rVert_2^2.
```

For two classes, the boundary satisfies:

```math
2z^{\mathsf T}
\left(
    \widehat\mu_1-
    \widehat\mu_0
\right)
=
\lVert\widehat\mu_1\rVert_2^2
-
\lVert\widehat\mu_0\rVert_2^2.
```

It is affine as a decision boundary, but its parameters are constrained by
sample means and the chosen Euclidean metric. Under a general affine transform,
the transformed prototypes are:

```math
\widehat\mu_c'
=
A\widehat\mu_c+a,
```

while the induced distance is:

```math
\lVert z'-\widehat\mu_c'\rVert_2^2
=
(z-\widehat\mu_c)^{\mathsf T}
A^{\mathsf T}A
(z-\widehat\mu_c).
```

Thus prototype classification is not equivalent to unrestricted linear
probing, despite having linear pairwise boundaries.

## 11. Prototype Estimation Variance

Assume class-conditional mean and covariance:

```math
\mathbb E[Z\mid Y=c]
=
\mu_c,
\qquad
\mathrm{Cov}(Z\mid Y=c)
=
\Sigma_c.
```

For independent support samples:

```math
\mathbb E
\left[
    \widehat\mu_c
\right]
=
\mu_c,
\qquad
\mathrm{Cov}
\left(
    \widehat\mu_c
\right)
=
\frac{\Sigma_c}{n_c}.
```

The expected squared estimation error is:

```math
\mathbb E
\left[
    \left\lVert
        \widehat\mu_c-
        \mu_c
    \right\rVert_2^2
\right]
=
\frac{\mathrm{tr}(\Sigma_c)}{n_c}.
```

Few-shot prototype evaluation therefore mixes at least three properties:

```math
\text{class separation},
\qquad
\text{within-class dispersion},
\qquad
\text{support-sampling variance}.
```

A single favorable support draw is not evidence of stable few-shot geometry.
Repeated patient-level support sampling is part of the estimand.

## 12. Mahalanobis Prototypes Recover Affine Covariance

Suppose a shared positive-definite covariance estimate is used:

```math
d_{\Sigma}(z,\mu_c)
=
(z-\mu_c)^{\mathsf T}
\Sigma^{-1}
(z-\mu_c).
```

Under `z'=Az+a`, transform covariance as:

```math
\Sigma'
=
A\Sigma A^{\mathsf T}.
```

Then:

```math
\begin{aligned}
d_{\Sigma'}(z',\mu_c')
&=
(z-\mu_c)^{\mathsf T}
A^{\mathsf T}
\left(
    A\Sigma A^{\mathsf T}
\right)^{-1}
A
(z-\mu_c)\\
&=
d_{\Sigma}(z,\mu_c).
\end{aligned}
```

The metric and representation must transform together. This illustrates a
general rule:

```math
\boxed{
\text{geometry belongs to the representation-metric pair.}
}
```

## 13. Zero-Shot Text Directions

Assume nonzero image and text embeddings. Let their normalized forms be:

```math
\bar z(x)
=
\frac{f_{\phi}(x)}
{\lVert f_{\phi}(x)\rVert_2},
\qquad
\bar t(q)
=
\frac{g_{\psi}(q)}
{\lVert g_{\psi}(q)\rVert_2}.
```

For class prompt set `\mathcal Q_c`, define a prompt ensemble direction:

```math
t_c
=
\mathrm{Norm}
\left(
    \sum_{q\in\mathcal Q_c}
    \omega_{cq}
    \bar t(q)
\right).
```

The weighted prompt sum is assumed nonzero. The logit temperature satisfies:

```math
\tau>0.
```

Zero-shot logits are:

```math
\ell_c(x)
=
\frac{
    \bar z(x)^{\mathsf T}t_c
}{\tau}.
```

The prediction is:

```math
\widehat y(x)
=
\arg\max_c
\ell_c(x).
```

This evaluates accessibility through text-defined directions. It does not test
the best affine boundary available in image space.

## 14. Joint Symmetry Of Image-Text Evaluation

For a shared orthogonal transformation `Q`:

```math
(Q\bar z)^{\mathsf T}(Qt_c)
=
\bar z^{\mathsf T}t_c.
```

Thus zero-shot logits are invariant to joint orthogonal rotation of both
modalities. Independent transformations generally break alignment:

```math
(Q_I\bar z)^{\mathsf T}(Q_Tt_c)
=
\bar z^{\mathsf T}
Q_I^{\mathsf T}Q_T
t_c,
```

which equals the original similarity only under special relations between
`Q_I` and `Q_T`.

The relevant object is therefore the joint image-text geometry, not either
encoder in isolation.

## 15. Prompt Ensembles Change The Classifier

Two prompt sets for the same verbal class can yield different directions:

```math
t_c^{(1)}
\neq
t_c^{(2)}.
```

The resulting margin between classes `a` and `b` is:

```math
m_{ab}(x)
=
\bar z(x)^{\mathsf T}
\left(
    t_a-t_b
\right).
```

Prompt perturbation changes the margin by:

```math
\Delta m_{ab}(x)
=
\bar z(x)^{\mathsf T}
\left[
    \Delta t_a-
    \Delta t_b
\right].
```

By Cauchy-Schwarz:

```math
\left|
    \Delta m_{ab}(x)
\right|
\leq
\left\lVert
    \Delta t_a-
    \Delta t_b
\right\rVert_2.
```

The absolute perturbation bound does not depend on the original margin. What
does depend on that margin is decision fragility. A sufficient condition for
the class order to remain unchanged is:

```math
\left|
    m_{ab}(x)
\right|
>
\left\lVert
    \Delta t_a-
    \Delta t_b
\right\rVert_2.
```

Small original margins are therefore easier to flip, not necessarily more
sensitive in absolute score change. PLIP, CONCH, KEEP, and TITAN zero-shot
evaluations must report prompt templates, synonym aggregation, temperature
handling, and any patch-to-slide readout.

## 16. Ranking Metrics Observe Order, Not Scale

Let a scalar score be:

```math
r
:
\mathcal X
\longrightarrow
\mathbb R.
```

For independent positive and negative examples, population AUROC is:

```math
\mathrm{AUROC}(r)
=
\mathbb P
\left(
    r(X^+)>r(X^-)
\right)
+
\frac{1}{2}
\mathbb P
\left(
    r(X^+)=r(X^-)
\right).
```

Let `g` be strictly increasing. Then:

```math
r(X^+)>r(X^-)
\quad\Longleftrightarrow\quad
g(r(X^+))>g(r(X^-)).
```

Therefore:

```math
\boxed{
\mathrm{AUROC}(g\circ r)
=
\mathrm{AUROC}(r).
}
```

Strictly increasing transformations generate a monotone invariance orbit:

```math
\mathrm{Orb}_{\mathrm{mon}}(r)
=
\left\{
    g\circ r
    :
    g
    \text{ is strictly increasing}
\right\}.
```

This orbit is contained in, but need not equal, the full equal-AUROC class:

```math
\mathrm{Orb}_{\mathrm{mon}}(r)
\subseteq
\left\{
    r'
    :
    \mathrm{AUROC}(r')
    =
    \mathrm{AUROC}(r)
\right\}.
```

Two scores can have the same AUROC while disagreeing on many individual
pairwise orderings.

It cannot determine whether scores are probabilities, logits, hazards, risk
ratios, or arbitrary monotone transforms.

## 17. Perfect Ranking Can Have Poor Probability Quality

Consider a balanced binary test population with a perfectly informative binary
feature:

```math
X=Y.
```

Model `A` predicts from `X`:

```math
\widehat p_A
=
\begin{cases}
0.9,&X=1,\\
0.1,&X=0,
\end{cases}
```

while model `B` predicts:

```math
\widehat p_B
=
\begin{cases}
0.6,&X=1,\\
0.4,&X=0.
\end{cases}
```

Both models have:

```math
\mathrm{AUROC}_A
=
\mathrm{AUROC}_B
=
1,
```

and threshold accuracy at one half:

```math
\mathrm{Acc}_A
=
\mathrm{Acc}_B
=
1.
```

Their Brier risks differ:

```math
\begin{aligned}
\mathrm{BS}_A
&=
\mathbb E
\left[
    (Y-\widehat p_A)^2
\right]
=
0.01,\\
\mathrm{BS}_B
&=
\mathbb E
\left[
    (Y-\widehat p_B)^2
\right]
=
0.16.
\end{aligned}
```

This toy example is deliberately simple. It proves that equal discrimination
and equal decisions do not imply equal probabilistic sharpness or calibration.

## 18. Calibration Is Conditional Probability Matching

A binary probability predictor is calibrated when:

```math
\mathbb P
\left(
    Y=1
    \mid
    \widehat p=p
\right)
=
p
```

for values in the support of `\widehat p`. Define calibration function:

```math
c(p)
=
\mathbb E
\left[
    Y
    \mid
    \widehat p=p
\right].
```

An integrated squared calibration error is:

```math
\mathcal E_{\mathrm{cal}}
=
\mathbb E
\left[
    \left(
        c(\widehat p)-
        \widehat p
    \right)^2
\right].
```

This population quantity should not be confused with a particular binned
expected-calibration-error estimator. Binning introduces partition-dependent
bias and variance.

Let event prevalence be:

```math
\pi
=
\mathbb E[Y].
```

Using the law of total variance, population Brier risk decomposes exactly as:

```math
\begin{aligned}
\mathbb E
\left[
    (Y-\widehat p)^2
\right]
&=
\pi(1-\pi)
-
\mathrm{Var}
\left(
    c(\widehat p)
\right)\\
&\qquad+
\mathbb E
\left[
    \left(
        c(\widehat p)-
        \widehat p
    \right)^2
\right].
\end{aligned}
```

The three terms are uncertainty, resolution, and reliability error. Thus a low
Brier score can arise from both useful separation and good calibration, whereas
AUROC isolates ordering.

## 19. Metric-Output Compatibility

Every metric requires a minimum output object:

| Metric | Prediction object consumed | Scalar functional summarized | Not identified by the scalar |
|---|---|---|---|
| accuracy | class decision or thresholded score | mean correctness at the selected decision rule | ranking away from threshold, calibration, individual decisions |
| AUROC | scalar class score | weighted positive-negative pair-order score | threshold utility, calibration, individual pair orderings |
| log loss | class-probability vector | mean negative log probability assigned to outcomes | the predictive vectors themselves or subgroup behavior |
| Brier score | class probability or survival probability | mean squared probability error at the evaluated target | calibration function, individual probabilities, unmeasured horizons |
| concordance | scalar risk score | weighted concordant-pair proportion | individual pair order, baseline survival, absolute risk |
| retrieval recall | ranked archive indices | mean target recovery within a cutoff | exact neighbor lists or archive-independent semantics |
| segmentation Dice | pixel or region mask | mean set-overlap functional | individual masks, boundary distance, slide-level context |

The metric is undefined or semantically mismatched when the model does not
export the required object. Converting one object into another introduces a
readout that must be reported.

## 20. Linear Centered Kernel Alignment

Representational-similarity analysis has been applied directly to computational
pathology foundation models in the
[Mishra-Lotter comparison](https://arxiv.org/abs/2509.15482). Any such result
depends on the chosen similarity statistic and comparison sample. Linear CKA is
one precise example. This comparison is used here as an external motivating
evaluation study, not as a substitute for a paper-specific model derivation.

Representation-similarity analysis compares two feature matrices evaluated on
the same `n` inputs:

```math
X\in\mathbb R^{n\times d_x},
\qquad
Y\in\mathbb R^{n\times d_y}.
```

Let centering matrix be:

```math
H
=
I_n-
\frac{1}{n}
\mathbf 1\mathbf 1^{\mathsf T}.
```

Centered Gram matrices are:

```math
K
=
HXX^{\mathsf T}H,
\qquad
L
=
HYY^{\mathsf T}H.
```

Linear centered kernel alignment is:

```math
\mathrm{CKA}(X,Y)
=
\frac{
    \langle K,L\rangle_F
}{
    \lVert K\rVert_F
    \lVert L\rVert_F
}.
```

This definition assumes that both centered Gram matrices have nonzero
Frobenius norm.

Equivalently:

```math
\mathrm{CKA}(X,Y)
=
\frac{
    \left\lVert
        X_c^{\mathsf T}Y_c
    \right\rVert_F^2
}{
    \left\lVert
        X_c^{\mathsf T}X_c
    \right\rVert_F
    \left\lVert
        Y_c^{\mathsf T}Y_c
    \right\rVert_F
},
```

where:

```math
X_c=HX,
\qquad
Y_c=HY.
```

CKA measures normalized alignment of sample-sample Gram structure. It does not
directly measure label information, calibration, or downstream utility.

## 21. CKA Invariance Group

For nondegenerate centered Gram matrices, nonzero scalars `a,c`, orthogonal
matrices `Q,R` of the corresponding feature dimensions, and constant feature
translations `b,d`, linear CKA satisfies the bivariate invariance:

```math
\mathrm{CKA}
\left(
    aXQ+\mathbf 1b^{\mathsf T},
    cYR+\mathbf 1d^{\mathsf T}
\right)
=
\mathrm{CKA}(X,Y).
```

Centering removes both translations. Orthogonal right multiplication leaves
each centered Gram matrix unchanged, while the two nonzero scalar factors
cancel through CKA normalization.

As a self-comparison special case, assume both representations have feature
dimension `d`. Let `Q` be a `d`-dimensional orthogonal matrix, `a` a nonzero
scalar, and `b` a constant feature translation:

```math
Y
=
aXQ
+
\mathbf 1b^{\mathsf T}.
```

Centering removes the translation:

```math
HY
=
aHXQ.
```

Then the centered Gram matrix satisfies:

```math
(HY)(HY)^{\mathsf T}
=
a^2
(HX)(HX)^{\mathsf T},
```

so:

```math
\mathrm{CKA}
\left(
    X,
    aXQ+\mathbf 1b^{\mathsf T}
\right)
=
1.
```

For a general invertible matrix `A`:

```math
Y
=
XA,
```

centering gives:

```math
Y_c
=
X_cA,
```

and hence:

```math
Y_cY_c^{\mathsf T}
=
X_cAA^{\mathsf T}X_c^{\mathsf T}.
```

This centered Gram matrix is not generally proportional to
`X_cX_c^{\mathsf T}`. Therefore:

```math
\mathrm{CKA}(X,XA)
<
1
```

can occur even though every unrestricted affine probe on `X` has an equivalent
probe on `XA`.

The consequence is precise:

```math
\boxed{
\text{low representational similarity does not imply different linear
transfer capacity.}
}
```

To make the gap quantitative, suppose the centered feature matrix satisfies:

```math
X_c^{\mathsf T}X_c
=
I_d.
```

For:

```math
A_{\kappa}
=
\mathrm{diag}
\left(
    \kappa,
    1,
    \ldots,
    1
\right),
\qquad
\kappa>0,
```

the linear CKA is:

```math
\mathrm{CKA}
\left(
    X,
    XA_{\kappa}
\right)
=
\frac{
    \kappa^2+d-1
}{
    \sqrt d
    \sqrt{
        \kappa^4+d-1
    }
}.
```

Hence:

```math
\lim_{\kappa\to\infty}
\mathrm{CKA}
\left(
    X,
    XA_{\kappa}
\right)
=
\frac{1}{\sqrt d}.
```

Every finite `\kappa` gives an invertible map, so the unrestricted affine-probe
family remains exactly unchanged. In high dimension, CKA can therefore be
small while affine transfer capacity is identical.

Conversely, high CKA on one finite sample does not prove equal transfer under a
new distribution or equal access to a particular label.

## 22. CKA Is Sample-Relative

Let evaluation inputs be drawn from distribution `P`, and define random
features:

```math
Z_1
=
f_1(X),
\qquad
Z_2
=
f_2(X).
```

Assume finite second moments. For `a,b` in `{1,2}`, define cross-covariance:

```math
\Sigma_{ab}^{P}
=
\mathbb E_P
\left[
    \left(
        Z_a-\mathbb E_P[Z_a]
    \right)
    \left(
        Z_b-\mathbb E_P[Z_b]
    \right)^{\mathsf T}
\right].
```

When both covariance norms in the denominator are nonzero, define population
linear CKA:

```math
\mathrm{CKA}_{P}(f_1,f_2)
=
\frac{
    \left\lVert
        \Sigma_{12}^{P}
    \right\rVert_F^2
}{
    \left\lVert
        \Sigma_{11}^{P}
    \right\rVert_F
    \left\lVert
        \Sigma_{22}^{P}
    \right\rVert_F
}.
```

The population covariance relation can change under a shifted distribution
`Q`:

```math
\mathrm{CKA}_{P}(f_1,f_2)
\neq
\mathrm{CKA}_{Q}(f_1,f_2).
```

For pathology encoders, the comparison sample must report:

```math
\text{organs},
\quad
\text{magnifications},
\quad
\text{stains},
\quad
\text{institutions},
\quad
\text{patch sampling},
\quad
\text{token readout}.
```

Comparing Virchow's concatenated class-token and patch-token mean against UNI's
class token, for example, compares both encoder geometry and exported-token
choice. Token construction is part of the evaluation contract.

## 23. Why Evaluation Rankings Can Reverse

Let the two induced scalar prediction families be:

```math
\mathcal F_1
=
\mathcal H_1\circ f_1,
\qquad
\mathcal F_2
=
\mathcal H_2\circ f_2.
```

Assume these families are contained in `L^2(P_X)`, with loss pseudometric:

```math
d_2(g,h)
=
\left(
    \mathbb E
    \left[
        (g(X)-h(X))^2
    \right]
\right)^{1/2}.
```

If there exists `g_1` in `\mathcal F_1` at positive distance from the closure of
`\mathcal F_2`:

```math
\delta_1
=
\inf_{h\in\mathcal F_2}
d_2(g_1,h)
>
0,
```

then the noiseless target `Y=g_1(X)` has best squared risks:

```math
\inf_{g\in\mathcal F_1}
\mathbb E[(Y-g(X))^2]
=
0,
```

and:

```math
\inf_{h\in\mathcal F_2}
\mathbb E[(Y-h(X))^2]
\geq
\delta_1^2.
```

If symmetrically there exists `g_2` in `\mathcal F_2` at positive distance from
the closure of `\mathcal F_1`, a second target reverses the ordering. Plain set
noncontainment is insufficient because an excluded function may still be
approximated arbitrarily well.

More generally, benchmark ordering is a partial relation indexed by protocol:

```math
f_1
\succ_{\mathfrak E_1}
f_2
```

does not imply:

```math
f_1
\succ_{\mathfrak E_2}
f_2.
```

A benchmark suite should expose these reversals rather than average them into
one context-free leaderboard number.

## 24. Metric Contradiction Matrix

| Observation | What follows | What does not follow |
|---|---|---|
| equal unregularized linear-probe risk | equal best risk within affine heads | equal neighborhoods, CKA, or calibration |
| equal retrieval metric value | equal aggregate metric under the evaluated memory and queries | equal neighbor lists, linear transfer, or external-memory retrieval |
| equal AUROC | equal aggregate pair-ordering score | equal individual pair orderings, threshold utility, or probability quality |
| equal concordance | equal weighted concordant-pair proportion | equal pairwise risk order, survival curves, or horizon calibration |
| high CKA | high normalized finite-sample Gram alignment | equal label sufficiency or external transfer |
| equal zero-shot accuracy | equal aggregate correctness for the chosen prompts and reader | equal decisions or image-only linear accessibility |
| equal patch accuracy | equal aggregate local correctness under one protocol | equal patch decisions or WSI performance after aggregation |
| equal WSI accuracy | equal aggregate correctness for one aggregator and dataset | equal decisions, patch localization, or spatial sufficiency |

## 25. Paper-Grounded Measurement Contracts

The detailed model derivations live in the paper-specific folder. This matrix
records only the exact object-to-score path needed to interpret an evaluation.

UNI and Virchow make the exported-token distinction explicit:

```math
h^{\mathrm{UNI}}(x)
=
u_0
\in
\mathbb R^{1024},
```

whereas Virchow exports:

```math
e^{\mathrm{Virchow}}(x)
=
\left[
    c(x)
    \,;
    \frac{1}{256}
    \sum_{r=1}^{256}
    p_r(x)
\right]
\in
\mathbb R^{2560}.
```

| Source | Exported object | Evaluation path | Fitted downstream components | Functional summarized |
|---|---|---|---|---|
| [HIPT](https://arxiv.org/abs/2206.02647) | pretrained local and regional representations | hierarchical tokens to task-trained WSI module | WSI task module | task-specific classification or survival functional |
| [RetCCL](https://pubmed.ncbi.nlm.nih.gov/36270093/) | patch vectors plus selected feature-spatial mosaic | query mosaic to diagnosis-aware source-slide archive ranking | nonparametric retrieval rule after encoder training | ranked retrieval functional |
| [UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/) | 1024-dimensional class token | frozen tile vectors to experiment-specific probe, prototype, or MIL reader | downstream reader only | reader-specific task functional |
| [Virchow](https://arxiv.org/abs/2309.07778) | 2560-dimensional class-token and local-token-mean concatenation | frozen tile vectors to experiment-specific task reader | downstream reader only | reader-specific task functional |
| [PLIP](https://pubmed.ncbi.nlm.nih.gov/37592105/) | normalized image-text vector | cosine prompt direction for zero-shot evaluation or learned affine probe | none for zero-shot; probe for supervised transfer | prompt-defined or affine task functional |
| [CONCH](https://arxiv.org/abs/2307.12914) | one-query contrastive vector; separate 256-query caption token set | one-query vector for retrieval and tile zero-shot; class-specific top-K aggregation for WSI zero-shot; 256 tokens for caption decoding | none for zero-shot; task reader or decoder when fitted | retrieval, zero-shot, or generation functional for the named path |
| [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w) | frozen tile vectors followed by contextual slide states | frozen tile encoder to fine-tuned LongNet to shallow ABMIL to task classifier in the described slide-task chain | LongNet, ABMIL, and classifier; ablations must be named separately | supervised task functional of the complete chain |
| [TITAN](https://www.nature.com/articles/s41591-025-03982-3) | one-query slide vector; separate 128-query generation token set | one-query vector to linear, prototype, prompt, archive, or Cox reader; 128 tokens to decoder | task-dependent reader or decoder | task, retrieval, concordance, or generation functional for the named path |
| [KEEP](https://arxiv.org/html/2412.13126v1) | knowledge-aligned tile and text vectors | tile image-text logits to hard class `\arg\max` to tile-class frequencies and tumor-ratio slide statistic | no downstream gradient in the zero-shot reader | hard-vote or tumor-ratio slide functional |

## 26. C/R/G/S/H Evaluation Placement

For evaluation, preserve the repository's five-part distinction:

```math
\mathfrak E
\longmapsto
\left(
    C_{\mathrm{eval}},
    R_{\mathrm{eval}},
    G_{\mathrm{eval}},
    S_{\mathrm{eval}},
    H_{\mathrm{eval}}
\right).
```

### 26.1 Context Operator

```math
C_{\mathrm{eval}}
\in
\left\{
    \text{none},
    \text{attention},
    \text{transformer},
    \text{graph},
    \text{state space},
    \text{cross-modal}
\right\}.
```

This is the interaction operator applied before export: no interaction for an
isolated patch probe, LongNet context for Prov-GigaPath, image-text interaction
for CONCH or TITAN. Ordinary post-hoc retrieval uses no additional context
operator.

### 26.2 Readout Operator

```math
R_{\mathrm{eval}}
\in
\left\{
    \text{class token},
    \text{mean},
    \text{attention sum},
    \text{prototype distances},
    \text{top-}K\text{ statistic},
    \text{vote histogram},
    \text{neighbor list},
    \text{query-token set}
\right\}.
```

This is the aggregation or export map. It must not be conflated with the final
task head.

### 26.3 Geometry

Treat evaluation geometry as a structured specification rather than as one
operator:

```math
G_{\mathrm{eval}}
=
\left(
    G_C,
    G_R,
    G_M
\right).
```

The three components have distinct roles:

```math
\begin{aligned}
G_C
&:
\text{coordinates, grid, graph, hierarchy, or cross-modal relation used by context},\\
G_R
&:
\text{Euclidean, cosine, Mahalanobis, normalization, archive, exclusions, and ties},\\
G_M
&:
\text{decision, rank, probability, survival, retrieval, or spatial-overlap relation}.
\end{aligned}
```

Unused components are set to `none`. This typing separates geometry consumed by
context, geometry used to construct a readout such as a neighbor list, and the
relation summarized by the final metric.

### 26.4 Supervision And Task Signal

```math
S_{\mathrm{eval}}
\in
\left\{
    \text{class labels},
    \text{censoring and event times},
    \text{few-shot support},
    \text{text prompts},
    \text{retrieval relevance},
    \text{no task supervision}
\right\}.
```

This is the signal that selects or fits the evaluation rule. It is distinct
from the surviving representation statistic, which is derived from context and
readout:

```math
T_{\mathrm{surv}}
=
R_{\mathrm{eval}}
\left(
    C_{\mathrm{eval}}
    \left(
        f_{\phi}(X);
        G_C
    \right);
    G_R
\right).
```

Examples of `T_{\mathrm{surv}}` include a class token, first moment, weighted
first moment, prototype-distance vector, top-K statistic, vote histogram,
metric neighborhood, or query-token sequence.

### 26.5 Task Head

```math
H_{\mathrm{eval}}
\in
\left\{
    \text{identity},
    \text{linear classifier},
    \text{logistic head},
    \text{prototype decision},
    \text{similarity argmax},
    \text{threshold decision},
    \text{hard vote},
    \text{nearest-neighbor decision},
    \text{Cox head},
    \text{hazard head},
    \text{decoder}
\right\}.
```

Linear, Cox, and decoder maps belong here, not in `R`. An evaluation claim is
legible only when all five components are named. The full transfer and
reporting protocol is derived in the
[foundation-model transfer-limits note](../paper_specific_derivations/10_foundation_model_transfer_and_readout_limits.md); this note adds the
evaluation-specific invariance group and scalar functional.

## 27. Bottom Line

An evaluation does not reveal the representation in full. It reveals the part
that survives a chosen context and readout, reaches a named task head, and is
then collapsed by a metric under a chosen geometry. The forward map is:

```math
\begin{aligned}
\widetilde Z
&=
C_{\mathrm{eval}}
\left(
    f_{\phi}(X);
    G_C
\right),\\
T_{\mathrm{surv}}
&=
R_{\mathrm{eval}}
\left(
    \widetilde Z;
    G_R
\right),\\
\widehat Y
&=
H_{\mathrm{eval}}
\left(
    T_{\mathrm{surv}}
\right).
\end{aligned}
```

The task signal `S_{\mathrm{eval}}` determines how applicable trainable
components or prompt-defined decisions are selected. Comparison is determined
by:

```math
\Gamma_{G_M,M}
\left(
    \widehat Y,
    Y;
    \mathcal D_{\mathrm{test}}
\right)
=
\widehat V.
```

Here `\Gamma_{G_M,M}` is the actual comparison functional. It is not assumed
to be a distance. Equivalently:

```math
\Gamma_{G_M,M}
:
\left(
    \widehat Y,
    Y
\right)
\longmapsto
\widehat V.
```

Linear probing quotients out invertible affine coordinates when optimization is
exact over an unrestricted affine head with compatible bias handling. This
holds for empirical or population risk; regularization, constraints, and
optimization can reduce the invariance. Retrieval preserves a narrower metric
geometry and depends on an archive. Zero-shot evaluation inspects text-defined
directions. Ranking metrics preserve order but discard probability scale. CKA
compares finite-sample Gram structure but does not identify task sufficiency.

The correct question is therefore not:

```text
Which pathology foundation model is best?
```

It is:

```text
Which model-context-readout-head-metric system preserves the statistic required
by the task, under the shift and sampling unit that the claim is supposed to
cover?
```
