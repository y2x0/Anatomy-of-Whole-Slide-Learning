# Foundation Model Transfer and Readout Limits

Primary sources:

- [UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/)
- [Virchow](https://arxiv.org/abs/2309.07778)
- [CONCH](https://arxiv.org/abs/2307.12914)
- [PLIP](https://pubmed.ncbi.nlm.nih.gov/37592105/)
- [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w)
- [TITAN](https://www.nature.com/articles/s41591-025-03982-3)
- [HIPT](https://arxiv.org/abs/2206.02647)
- [clinical benchmark of public pathology foundation models](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003829/)
- [paper-specific derivations in this folder](README.md)

The transfer question is not merely whether one frozen encoder obtains a
higher score than another. A downstream result combines at least four
mathematical effects:

```text
information retained by the pretrained representation;
information accessible to the chosen readout class;
information estimable from the finite labeled patient cohort;
stability of all three under deployment shift.
```

This note separates those effects. It also distinguishes the transfer claims
made by patch encoders, slide encoders, vision-language models, and
memory-dependent retrieval systems.

## 1. The Transfer Experiment Is A Composite Map

Let a slide object be:

```math
X_i
=
\left\{
    (x_{ij},c_{ij})
    :
    j=1,\ldots,n_i
\right\},
```

where `x` is a tissue patch and `c` can contain coordinates, scale, region,
or acquisition metadata.

A pretrained map produces an exported object:

```math
Z_i
=
\Phi_{\phi}(X_i).
```

The codomain of `Phi` depends on the model:

```math
\begin{aligned}
\text{patch encoder:}
&\quad
Z_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}\in\mathbb R^d,
\\
\text{slide encoder:}
&\quad
Z_i
=
z_i\in\mathbb R^D,
\\
\text{multitoken slide encoder:}
&\quad
Z_i
=
[z_{i1},\ldots,z_{im}]
\in
\mathbb R^{m\times D},
\\
\text{vision-language encoder:}
&\quad
Z_i
=
(z_i,t_i)
\in
\mathbb S^{D-1}\times\mathbb S^{D-1}.
\end{aligned}
```

The downstream learner is another map:

```math
\widehat y_i
=
h_{\widehat\eta}(Z_i),
\qquad
h_{\widehat\eta}
\in
\mathcal H.
```

For a patch encoder, `h` generally factorizes into context, readout, and task
head:

```math
\widehat y_i
=
\mathcal H_{\omega}
\left(
    \mathcal R_{\psi}
    \left(
        \mathcal C_{\psi}
        \left(
            \{f_{\phi}(x_{ij}),c_{ij}\}_{j=1}^{n_i}
        \right)
    \right)
\right).
```

Consequently, a benchmark score is a property of the full tuple:

```math
\left(
P_{\mathrm{pre}},
\Phi_{\phi},
P_{\mathrm{task}},
\mathcal H,
\mathcal A,
\mathcal M
\right),
```

where `A` is the downstream fitting procedure and `M` is the evaluation
metric. It is not an intrinsic scalar property of the encoder alone.

## 2. Exact Source-Risk Decomposition

Let `P_s` be the labeled source distribution and let `ell` be the downstream
loss. Define:

```math
\mathcal R_s(g\circ\Phi)
=
\mathbb E_{(X,Y)\sim P_s}
\left[
    \ell
    \left(
        Y,
        g(\Phi(X))
    \right)
\right].
```

The Bayes risk with the raw slide is:

```math
\mathcal R_s^{X}
=
\inf_{f\ \mathrm{measurable}}
\mathbb E_s
\left[
    \ell(Y,f(X))
\right].
```

The Bayes risk after representation compression is:

```math
\mathcal R_s^{Z}
=
\inf_{g\ \mathrm{measurable}}
\mathbb E_s
\left[
    \ell(Y,g(Z))
\right],
\qquad
Z=\Phi(X).
```

For a restricted downstream class:

```math
\mathcal R_s^{\mathcal H}
=
\inf_{h\in\mathcal H}
\mathcal R_s(h\circ\Phi).
```

If training returns `h_hat`, its source excess risk decomposes exactly:

```math
\boxed{
\begin{aligned}
\mathcal R_s(\widehat h\circ\Phi)
-
\mathcal R_s^{X}
&=
\underbrace{
\left(
\mathcal R_s^{Z}
-
\mathcal R_s^{X}
\right)
}_{\text{representation information gap}}
\\
&\quad+
\underbrace{
\left(
\mathcal R_s^{\mathcal H}
-
\mathcal R_s^{Z}
\right)
}_{\text{readout approximation gap}}
\\
&\quad+
\underbrace{
\left(
\mathcal R_s(\widehat h\circ\Phi)
-
\mathcal R_s^{\mathcal H}
\right)
}_{\text{finite-sample and optimization gap}}.
\end{aligned}
}
```

The first gap is caused by `Phi`. The second is caused by the chosen probe or
readout class. The third is caused by finite labeled data, regularization,
optimization, and model selection.

These gaps answer different questions:

```text
representation gap:
    did pretraining discard a task-relevant distinction?

readout gap:
    is that distinction present but inaccessible to this head?

estimation gap:
    could this head express the distinction but fail to learn it from this
    cohort and training procedure?
```

A single linear-probe score cannot identify which gap dominates.

## 3. Log-Loss Gives An Information Identity

Assume classification with conditional class distribution:

```math
p(y\mid x).
```

Under logarithmic loss, the Bayes risk from the raw slide is conditional
entropy:

```math
\mathcal R_{\log}^{X}
=
H(Y\mid X).
```

The Bayes risk from the frozen representation is:

```math
\mathcal R_{\log}^{Z}
=
H(Y\mid Z).
```

Because `Z` is produced from `X`, the representation gap is:

```math
\boxed{
\mathcal R_{\log}^{Z}
-
\mathcal R_{\log}^{X}
=
H(Y\mid Z)
-
H(Y\mid X)
=
I(Y;X\mid Z).
}
```

Thus the representation is task-sufficient on the evaluated distribution if
and only if:

```math
I(Y;X\mid Z)
=
0.
```

Equivalently:

```math
Y
\perp
X
\mid
Z.
```

This is a task-conditional statement. It does not require the representation
to preserve every pixel or to be invertible. It requires every distinction in
`X` that changes the conditional label distribution to remain available in
`Z`.

The corresponding collision criterion is:

```math
\Phi(X)=\Phi(X')
\Longrightarrow
P_s(Y\mid X)
=
P_s(Y\mid X')
```

almost surely on the source support.

## 4. Square Loss Gives A Conditional-Variance Identity

For a real-valued target, define:

```math
m_X(X)
=
\mathbb E[Y\mid X],
\qquad
m_Z(Z)
=
\mathbb E[Y\mid Z].
```

Because `Z` is a function of `X`:

```math
m_Z(Z)
=
\mathbb E[m_X(X)\mid Z].
```

Under squared loss, the representation gap is:

```math
\boxed{
\mathcal R_{2}^{Z}
-
\mathcal R_{2}^{X}
=
\mathbb E
\left[
    \left(
        m_X(X)-m_Z(Z)
    \right)^2
\right]
=
\mathbb E
\left[
    \mathrm{Var}
    \left(
        m_X(X)
        \mid
        Z
    \right)
\right].
}
```

The gap is zero only when the optimal raw-slide predictor is constant within
each representation fiber:

```math
\Phi^{-1}(z)
=
\{X:\Phi(X)=z\}.
```

This identity makes the phrase "missing statistic" precise. A missing
statistic is task-relevant variation of `m_X` inside one fiber of `Phi`.

## 5. A Readout Cannot Create Task Information

Whenever the transfer chain is Markov:

```math
Y
\longrightarrow
X
\longrightarrow
Z
\longrightarrow
\widehat Y,
```

the data-processing inequality gives:

```math
I(Y;\widehat Y)
\le
I(Y;Z)
\le
I(Y;X).
```

An attention head, graph readout, prototype classifier, or transformer can
re-express information in `Z`; it cannot reconstruct task information that is
absent from `Z`.

For a stored patch-feature pipeline, this boundary is literal:

```math
\{x_{ij}\}
\xrightarrow{\ \Phi\ }
\{h_{ij}\}
\xrightarrow{\ \mathcal R\ }
z_i.
```

If raw patches are discarded after feature extraction, every downstream model
is measurable with respect to the sigma-field generated by the stored
features:

```math
\widehat Y_i
\in
\sigma
\left(
    \{h_{ij},c_{ij}\}_{j=1}^{n_i}
\right).
```

Fine-tuning cannot cross this boundary unless the raw patches are available
and the encoder is actually rerun.

## 6. Linear Probing Tests One Hypothesis Class

For a frozen vector:

```math
z_i
\in
\mathbb R^D,
```

a multiclass linear probe is:

```math
p_{\eta}(Y=c\mid z_i)
=
\frac{
    \exp
    \left(
        w_c^{\mathsf T}z_i+b_c
    \right)
}{
    \sum_{r=1}^{C}
    \exp
    \left(
        w_r^{\mathsf T}z_i+b_r
    \right)
}.
```

The probe family is:

```math
\mathcal H_{\mathrm{lin}}
=
\left\{
z\mapsto Wz+b
\right\}.
```

High performance implies that the task is approximately accessible through
affine decision boundaries under the chosen source distribution, label budget,
regularization, and metric.

Low performance has at least three explanations:

```math
\begin{aligned}
&I(Y;X\mid Z)>0,
\\
&\mathcal R_s^{\mathcal H_{\mathrm{lin}}}
>
\mathcal R_s^{Z},
\\
&\mathcal R_s(\widehat h_{\mathrm{lin}})
>
\mathcal R_s^{\mathcal H_{\mathrm{lin}}}.
\end{aligned}
```

These mean, respectively:

```text
the representation lacks the signal;
the signal is present but nonlinear;
the finite cohort or fitting procedure failed to estimate the linear rule.
```

Calling all three "poor representation quality" is mathematically unjustified.

## 7. Probe Strength Changes The Question

Let two downstream families satisfy:

```math
\mathcal H_1
\subseteq
\mathcal H_2.
```

Then their optimal source risks satisfy:

```math
\mathcal R_s^{\mathcal H_2}
\le
\mathcal R_s^{\mathcal H_1}.
```

This monotonicity is an approximation statement. Finite-sample test risk need
not be monotone because the larger family can have a larger estimation gap.

For example:

```math
\mathcal H_{\mathrm{lin}}
\subset
\mathcal H_{\mathrm{MLP}}
\subset
\mathcal H_{\mathrm{universal}}
```

under architectures that contain the smaller maps as special cases.

A very weak probe underestimates accessible information. A very strong probe
can learn the downstream task nearly from scratch and obscure whether the
pretrained geometry was useful. Therefore probe strength is an experimental
axis, not a nuisance detail.

## 8. Invertible Reparameterization Separates Information From Geometry

Let the exported representation be transformed by an invertible affine map:

```math
z'
=
Az+b,
\qquad
A\in\mathbb R^{D\times D},
\qquad
\det(A)\ne0.
```

The two representations contain the same measurable information because:

```math
z
=
A^{-1}(z'-b).
```

They also have the same optimal linear-probe risk. Every score:

```math
w^{\mathsf T}z+\beta
```

can be written in the new coordinates as:

```math
w'^{\mathsf T}z'+\beta',
\qquad
w'
=
A^{-\mathsf T}w,
\qquad
\beta'
=
\beta-w^{\mathsf T}A^{-1}b.
```

Cosine and Euclidean retrieval are not invariant to a general invertible map.
For Euclidean distance:

```math
\|Az_a-Az_b\|_2^2
=
(z_a-z_b)^{\mathsf T}
A^{\mathsf T}A
(z_a-z_b).
```

The metric is unchanged up to one global scale only when:

```math
A^{\mathsf T}A
=
\gamma I
```

for positive `gamma`.

Therefore:

```text
linear-probe equivalence does not imply retrieval equivalence;
retrieval equivalence does not imply calibration equivalence;
information equivalence does not imply optimization equivalence.
```

Centering, whitening, block normalization, and feature concatenation are part
of the evaluation contract.

## 9. Retrieval Is A Representation Plus A Metric Plus A Memory

Let the query representation be `z_q` and the archive be:

```math
\mathcal M
=
\{(z_r,y_r,p_r)\}_{r=1}^{M},
```

where `p_r` is a patient identifier.

For a dissimilarity `d`, the top-`K` archive indices are:

```math
\mathcal N_K(q)
=
\underset{
    A\subseteq\{1,\ldots,M\},
    |A|=K,
    p_r\ne p_q
}{\arg\min}
\sum_{r\in A}
d(z_q,z_r).
```

A retrieval result is thus a function of:

```math
\mathcal N_K
=
\mathcal N_K
\left(
    \Phi,
    d,
    \mathcal M,
    K,
    \mathrm{deduplication},
    \mathrm{patient\ exclusion}
\right).
```

Recall at `K` can be written as:

```math
\mathrm{Recall@K}
=
\frac{1}{Q}
\sum_{q=1}^{Q}
\mathbf 1
\left{
    \exists r\in\mathcal N_K(q)
    :
    y_r=y_q
\right}.
```

Changing the archive can change the score while keeping the encoder fixed.
Changing the metric can change the score while preserving all representation
information. A retrieval benchmark is not an encoder-only assay.

TITAN makes this distinction especially visible because its one-query slide
vector supports cosine-style slide-report and slide-slide retrieval, while its
128-query token set supports report generation. Those are different surviving
statistics and different evaluation functions.

## 10. Zero-Shot Transfer Tests Text-Defined Directions

Let normalized slide and class-text embeddings be:

```math
u_i
=
\frac{\Phi_{\mathrm{img}}(X_i)}
{\|\Phi_{\mathrm{img}}(X_i)\|_2},
\qquad
v_c
=
\frac{\Phi_{\mathrm{text}}(q_c)}
{\|\Phi_{\mathrm{text}}(q_c)\|_2}.
```

The zero-shot classifier is:

```math
\widehat y_i
=
\arg\max_c
u_i^{\mathsf T}v_c.
```

This tests whether a task label can be expressed by a prompt direction already
aligned with the visual geometry. It does not test the same class as a learned
linear probe:

```math
\underbrace{
\{v_c:\ v_c=\Phi_{\mathrm{text}}(q_c)\}
}_{\text{text-constrained directions}}
\subseteq
\underbrace{
\mathbb R^D
}_{\text{learnable linear directions}}.
```

For prompt templates `q_{c1},...,q_{cR}`, an ensemble direction is often:

```math
\overline v_c
=
\frac{
    \sum_{r=1}^{R}v_{cr}
}{
    \left\|
        \sum_{r=1}^{R}v_{cr}
    \right\|_2
}.
```

Prompt ensembling changes the classifier even though the visual encoder is
unchanged. A zero-shot score should therefore report prompts, aggregation,
normalization, and whether class text was written before viewing test results.

The similarity score is not automatically calibrated:

```math
u_i^{\mathsf T}v_c
\not\equiv
P(Y_i=c\mid X_i).
```

## 11. Patch Encoders Still Require A Slide Readout

UNI and Virchow export patch-level vectors. For encoder `b`:

```math
h_{ij}^{(b)}
=
f_{\phi_b}(x_{ij}).
```

Mean pooling followed by a linear head gives:

```math
z_i^{\mathrm{mean}}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}h_{ij},
```

```math
s_i^{\mathrm{mean}}
=
w^{\mathsf T}z_i^{\mathrm{mean}}+b
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
w^{\mathsf T}h_{ij}
+b.
```

This is average additive patch evidence. It cannot represent a label that
depends only on patch variance when the means match.

Attention MIL gives:

```math
e_{ij}
=
w_a^{\mathsf T}
\left[
    \tanh(V_a h_{ij})
    \odot
    \sigma(U_a h_{ij})
\right],
```

```math
\alpha_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{r=1}^{n_i}\exp(e_{ir})},
\qquad
z_i^{\mathrm{attn}}
=
\sum_{j=1}^{n_i}
\alpha_{ij}h_{ij}.
```

The attention statistic is a learned weighted first moment. It is more
adaptive than the unweighted mean but still lies in the convex hull of the
instance vectors:

```math
z_i^{\mathrm{attn}}
\in
\mathrm{conv}
\left(
    \{h_{ij}\}_{j=1}^{n_i}
\right).
```

A contextual MIL model first updates each patch:

```math
\widetilde h_{ij}
=
\mathcal C_{\psi}
\left(
    h_{ij},
    \{h_{ir},c_{ir}\}_{r=1}^{n_i}
\right),
```

and then pools:

```math
z_i
=
\mathcal R_{\psi}
\left(
    \{\widetilde h_{ij}\}_{j=1}^{n_i}
\right).
```

Comparing UNI or Virchow under different slide readers compares composite
systems. The patch encoder is held fixed, but the accessible slide statistic
changes.

## 12. Readout Collision Theorem

Let `T(X)` be the object supplied to a downstream readout and let every model in
the evaluated family factor through `T`:

```math
h(X)
=
g(T(X)),
\qquad
g\in\mathcal G.
```

If two slides collide under `T`:

```math
T(X_A)
=
T(X_B),
```

then every evaluated readout produces the same output:

```math
\forall g\in\mathcal G,
\qquad
g(T(X_A))
=
g(T(X_B)).
```

Examples include:

```text
mean collision:
    equal patch-feature first moments;

set collision:
    equal feature multisets but different coordinates;

top-K collision:
    equal upper score order statistics but different prevalence;

single-query collision:
    equal pooled slide token but different discarded token sets;

retrieval collision:
    equal neighbor ranking under one archive and metric.
```

The theorem is elementary, but it prevents a common evaluation mistake: no
amount of downstream optimization can separate examples already identified by
the evaluated transfer object.

## 13. Slide Encoders Change The Transfer Boundary

Prov-GigaPath produces contextual tile embeddings and a slide-level object:

```math
\{x_{ij},c_{ij}\}
\xrightarrow{\ f_{\phi}\ }
\{h_{ij},c_{ij}\}
\xrightarrow{\ \mathrm{LongNet}_{\psi}\ }
\{q_{ij}\}
\xrightarrow{\ \mathcal R\ }
z_i.
```

TITAN produces a one-query vector and a 128-query token sequence:

```math
z_i^{(1)}
=
\mathrm{Pool}_{1}(H_i),
\qquad
Z_i^{(128)}
=
\mathrm{Pool}_{128}(H_i).
```

A linear probe on `z_i^(1)` tests whether the task is linearly accessible from
one pretrained slide statistic. It is not equivalent to:

```text
fine-tuning the slide encoder;
training ABMIL on the underlying CONCH patch vectors;
decoding from the 128-query token set;
retrieving from an external archive;
zero-shot classification from text prompts.
```

Each path changes `H`, the readout, or both.

## 14. Frozen Transfer And Fine-Tuning Are Different Estimands

For frozen transfer:

```math
\nabla_{\phi}
\mathcal L_{\mathrm{task}}
=
0.
```

Only downstream parameters change:

```math
\widehat\eta
=
\arg\min_{\eta}
\frac{1}{N}
\sum_{i=1}^{N}
\ell
\left(
    y_i,
    h_{\eta}(\Phi_{\phi}(X_i))
\right).
```

For fine-tuning:

```math
(\widehat\phi,\widehat\eta)
=
\arg\min_{\phi,\eta}
\frac{1}{N}
\sum_{i=1}^{N}
\ell
\left(
    y_i,
    h_{\eta}(\Phi_{\phi}(X_i))
\right)
+
\lambda
\Omega
\left(
    \phi,\phi_0
\right).
```

The regularizer can penalize representation drift:

```math
\Omega(\phi,\phi_0)
=
\|\phi-\phi_0\|_2^2
```

or function drift:

```math
\Omega(\phi,\phi_0)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
    \Phi_{\phi}(X_i)
    -
    \Phi_{\phi_0}(X_i)
\right\|_2^2.
```

Fine-tuning can reduce a representation gap on the labeled source task because
it changes `Phi`. It also enlarges the estimation problem and can destroy
pretrained invariances on a small cohort.

An observed fine-tuning gain is:

```math
\Delta_{\mathrm{FT}}
=
\mathcal M
\left(
    \widehat h_{\mathrm{FT}}
    \circ
    \widehat\Phi
\right)
-
\mathcal M
\left(
    \widehat h_{\mathrm{frozen}}
    \circ
    \Phi_0
\right).
```

It is a property of initialization, trainable parameter set, optimizer,
regularization, cohort size, and metric. It is not a pure estimate of
pretraining quality.

## 15. Parameter Count And Labeled-Patient Complexity

A `C`-class linear probe on a `D`-dimensional vector has:

```math
C(D+1)
```

parameters before identifiability constraints. A two-layer head with hidden
width `m` has order:

```math
\mathcal O
\left(
Dm+mC
\right)
```

parameters. A trainable MIL or slide transformer adds parameters and repeated
patch-level computations, but the independent labeled units remain patients
or slides, not the number of patch tokens.

For a bounded vector and bounded linear weight:

```math
\|z_i\|_2
\le
B,
\qquad
\|w\|_2
\le
W,
```

a Lipschitz-loss linear-probe generalization term has the schematic scale:

```math
\mathcal O
\left(
    \frac{LBW}{\sqrt{N_{\mathrm{ind}}}}
    +
    \sqrt{
        \frac{\log(1/\delta)}
        {N_{\mathrm{ind}}}
    }
\right),
```

where `N_ind` is the number of independent sampling units. The expression is a
capacity scaling law, not a replacement for a task-specific bound.

If one patient contributes `m` correlated slides or patches with common
variance and intra-patient correlation `rho`, the variance inflation factor is:

```math
\mathrm{DEFF}
=
1+(m-1)\rho.
```

The corresponding effective sample size is:

```math
N_{\mathrm{eff}}
=
\frac{Pm}
{1+(m-1)\rho},
```

where `P` is the number of patients. When `rho` is near one:

```math
N_{\mathrm{eff}}
\approx
P,
```

not `P m`.

This is why a patch-random split can make a frozen feature look transferable
by placing nearly duplicated patient morphology in train and test.

## 16. Few-Shot Transfer Measures Prototype Estimation

For class `c`, let the frozen slide vectors have population mean and covariance:

```math
\mu_c
=
\mathbb E[Z\mid Y=c],
\qquad
\Sigma_c
=
\mathrm{Cov}(Z\mid Y=c).
```

With `K` independent support slides, the empirical prototype is:

```math
\widehat\mu_c
=
\frac{1}{K}
\sum_{r=1}^{K}
z_{cr}.
```

Its mean-squared estimation error is:

```math
\mathbb E
\left[
    \|\widehat\mu_c-\mu_c\|_2^2
\right]
=
\frac{
    \mathrm{tr}(\Sigma_c)
}{K}.
```

A nearest-prototype rule is:

```math
\widehat y(z)
=
\arg\min_c
\left\|
    \widetilde z
    -
    \widetilde\mu_c
\right\|_2^2,
```

where the tildes denote the reported centering and normalization convention.

Few-shot performance combines class separation with prototype variance. In a
high-dimensional anisotropic feature space, small `K` can make the prototype
direction unstable even when a full-data linear probe performs well.

Repeated support sampling estimates this instability. TITAN's few-shot
evaluation repeats random support selection and compares prototype and linear
readouts; the repetition is part of the estimand, not decorative error bars.

## 17. Survival Transfer Tests A Risk-Head Family

For one frozen slide vector, a linear Cox head is:

```math
\eta_i
=
\beta^{\mathsf T}z_i,
```

```math
\lambda(t\mid z_i)
=
\lambda_0(t)
\exp(\eta_i).
```

The negative Cox partial log-likelihood is:

```math
\mathcal L_{\mathrm{Cox}}
=
-
\sum_{i:\delta_i=1}
\left[
    \eta_i
    -
    \log
    \sum_{r\in\mathcal R(t_i)}
    \exp(\eta_r)
\right].
```

This evaluates whether one scalar proportional-hazards ordering is linearly
accessible from the frozen representation. It does not establish that the
embedding supports:

```text
nonproportional hazards;
horizon-specific hazard crossing;
competing risks;
calibrated absolute survival without a baseline estimate;
patch-level survival attribution.
```

TITAN reports a linear Cox evaluation on its single slide embedding with
site-preserved stratification. That is a precise and useful transfer test, but
its mathematical scope is the linear proportional-hazards family.

The TITAN methods state that the Cox regularization value is selected as the
one producing the best average test metric across the five folds. If the same
fold outcomes are used both to select the regularization value and to report
performance, model selection can make the reported estimate optimistic. Let
the candidate regularization values be indexed by `r`, with noisy fold-average
metric estimates:

```math
\widehat M_r
=
M_r
+
\varepsilon_r.
```

Selecting:

```math
\widehat r
=
\arg\max_r
\widehat M_r
```

creates the winner's-curse inequality:

```math
\mathbb E
\left[
    \max_r\widehat M_r
\right]
\ge
\max_r
\mathbb E
\left[
    \widehat M_r
\right].
```

Nested selection or an untouched validation layer is needed to remove this
specific reuse. This is an evaluation-protocol limitation, not a claim that the
TITAN representation lacks survival information.

For several slides per patient, an additional patient readout is needed:

```math
z_p
=
\mathcal R_{\mathrm{patient}}
\left(
    \{z_{ps}\}_{s=1}^{m_p}
\right).
```

Selecting the largest slide, averaging slide vectors, and taking maximum risk
are different patient statistics.

## 18. Ranking Is Not Calibration

For binary classification, AUROC depends only on ordering of scores:

```math
\mathrm{AUROC}
=
\Pr
\left(
    s(Z^+)>s(Z^-)
\right)
+
\frac{1}{2}
\Pr
\left(
    s(Z^+)=s(Z^-)
\right).
```

Every strictly increasing transform preserves AUROC:

```math
g\ \mathrm{strictly\ increasing}
\Longrightarrow
\mathrm{AUROC}(g\circ s)
=
\mathrm{AUROC}(s).
```

The probabilities can nevertheless be badly calibrated. The Brier score is:

```math
\mathrm{BS}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
    p_i-y_i
\right)^2.
```

For survival, concordance similarly measures ordering rather than absolute
survival calibration. Therefore a foundation representation can transfer well
for ranking while requiring substantial recalibration at a new institution.

An evaluation matrix should separate:

```math
\left{
\text{discrimination},
\text{calibration},
\text{decision utility},
\text{external stability}
\right}.
```

## 19. Deployment Shift Adds Another Exact Decomposition

Let `P_t` be the target distribution. Target excess risk can be decomposed
using the same representation and readout class:

```math
\mathcal R_t(\widehat h\circ\Phi)
-
\mathcal R_t^{X}
=
\left(
\mathcal R_t^{Z}-\mathcal R_t^{X}
\right)
+
\left(
\mathcal R_t^{\mathcal H}-\mathcal R_t^{Z}
\right)
+
\left(
\mathcal R_t(\widehat h\circ\Phi)-\mathcal R_t^{\mathcal H}
\right).
```

The final term can be expanded around source risk:

```math
\begin{aligned}
\mathcal R_t(\widehat h\circ\Phi)
-
\mathcal R_t^{\mathcal H}
&=
\underbrace{
\left(
\mathcal R_t(\widehat h\circ\Phi)
-
\mathcal R_s(\widehat h\circ\Phi)
\right)
}_{\text{risk shift for the fitted head}}
\\
&\quad+
\underbrace{
\left(
\mathcal R_s(\widehat h\circ\Phi)
-
\mathcal R_s^{\mathcal H}
\right)
}_{\text{source estimation and optimization}}
\\
&\quad+
\underbrace{
\left(
\mathcal R_s^{\mathcal H}
-
\mathcal R_t^{\mathcal H}
\right)
}_{\text{change in best achievable class risk}}.
\end{aligned}
```

The individual shift terms need not be nonnegative, but the identity separates
the mechanisms.

Under covariate shift in representation space:

```math
P_s(Y\mid Z)
=
P_t(Y\mid Z),
\qquad
P_s(Z)
\ne
P_t(Z),
```

target risk can be written with importance weights when support overlaps:

```math
\mathcal R_t(h)
=
\mathbb E_s
\left[
    \frac{p_t(Z)}{p_s(Z)}
    \ell(Y,h(Z))
\right].
```

If:

```math
P_s(Y\mid Z)
\ne
P_t(Y\mid Z),
```

reweighting the feature marginal is insufficient. This conditional shift can
occur when a stain, scanner, report template, or institution shortcut changes
its relation to disease.

## 20. Five Transfer Counterexamples

### 20.1 Missing Statistic

Let independent bits be:

```math
U,V
\sim
\mathrm{Bernoulli}
\left(
    \frac{1}{2}
\right),
\qquad
X=(U,V),
\qquad
Y=V.
```

Let the representation keep only:

```math
Z=U.
```

Then:

```math
I(Y;Z)=0,
\qquad
H(Y\mid Z)=1\ \mathrm{bit},
\qquad
H(Y\mid X)=0.
```

No readout can recover `V` from `Z`. This is an encoder information failure.

### 20.2 Readout-Limited XOR

Let:

```math
U,V
\in
\{-1,+1\},
\qquad
Z=(U,V),
\qquad
Y
=
\mathbf 1\{UV=1\}.
```

The representation contains the target exactly:

```math
Y
=
\mathbf 1\{z_1z_2>0\}.
```

No affine decision boundary separates the two diagonal positive points from
the two diagonal negative points. A quadratic readout is perfect. This is a
readout approximation failure, not a representation failure.

### 20.3 Mean-Pooling Collision

Consider one-dimensional patch bags:

```math
H_A
=
\{-1,+1\},
\qquad
H_B
=
\{0,0\}.
```

Their means agree:

```math
\frac{-1+1}{2}
=
0
=
\frac{0+0}{2}.
```

Their second moments differ:

```math
\frac{(-1)^2+1^2}{2}
=
1,
\qquad
\frac{0^2+0^2}{2}
=
0.
```

A mean reader cannot classify a variance-defined task even though the patch
features contain the distinction.

### 20.4 Invertible Metric Rank Reversal

Let a query and two archive points be:

```math
q=(0,0),
\qquad
z_1=(1,0),
\qquad
z_2=(0,2).
```

Under the original Euclidean metric:

```math
\|z_1-q\|_2
=
1
<
2
=
\|z_2-q\|_2.
```

Apply the invertible map:

```math
A
=
\begin{bmatrix}
3&0\\
0&1
\end{bmatrix}.
```

Then:

```math
\|Az_1-Aq\|_2
=
3
>
2
=
\|Az_2-Aq\|_2.
```

The information is unchanged, but the nearest neighbor flips. This is a metric
geometry failure.

### 20.5 Shortcut Reversal Under Shift

Let `Z` encode a scanner indicator. On the source institution:

```math
P_s(Y=Z)=1.
```

On the target institution:

```math
P_t(Y=1-Z)=1.
```

A source-trained linear head can be perfect on `P_s` and perfectly wrong on
`P_t`. The source representation is predictive but not stable under the target
mechanism.

These examples isolate five different failures that one average benchmark
score would collapse.

## 21. Paper-Specific Transfer Contracts

### 21.1 UNI And Virchow

The exported object is a patch vector. The downstream slide result therefore
depends on the chosen WSI reader:

```math
X_i
\xrightarrow{\ f_{\mathrm{UNI/Virchow}}\ }
\{h_{ij}\}
\xrightarrow{\ \mathcal R_{\mathrm{WSI}}\ }
z_i
\xrightarrow{\ \mathcal H_{\mathrm{task}}\ }
\widehat y_i.
```

The original UNI feature and Virchow feature also expose different crop-level
statistics. UNI exports a class token for the original model; Virchow
concatenates a class token with the first moment of local patch tokens. Their
dimensions and normalization conventions should not be silently mixed.

The clean transfer comparison holds constant:

```text
patch coordinates and tessellation;
slide readout architecture;
label split and patient grouping;
feature normalization;
optimizer and regularization;
metric and confidence interval procedure.
```

Otherwise the comparison is between pipelines, not only encoders.

### 21.2 CONCH And PLIP

Zero-shot transfer uses text-defined directions:

```math
\widehat y_i^{\mathrm{zero}}
=
\arg\max_c
u_i^{\mathsf T}v_c.
```

Linear transfer learns directions from labels:

```math
\widehat y_i^{\mathrm{lin}}
=
\arg\max_c
\left(
    w_c^{\mathsf T}u_i+b_c
\right).
```

CONCH WSI zero-shot transfer can additionally use class-specific top-`K` tile
scores. That readout preserves high-scoring order statistics rather than slide
prevalence or spatial arrangement. ABMIL transfer learns another weighted
first moment. The three scores test distinct semantic and aggregation
contracts.

### 21.3 HIPT

HIPT exports nested class-token summaries across fixed spatial scales. Frozen
lower stages and a task-trained upper stage test whether local and regional
summaries retained the target distinction. A failure can arise at any class
token bottleneck or from mismatch between fixed windows and the tissue region
that determines the label.

### 21.4 Prov-GigaPath

Prov-GigaPath changes both representation scale and downstream transfer path:

```math
\text{tile DINOv2}
\longrightarrow
\text{LongNet slide context}
\longrightarrow
\text{softmax attention}
\longrightarrow
\text{task head}.
```

Its task-specific evaluation freezes the pretrained tile encoder, fine-tunes
the LongNet slide encoder, and trains a shallow ABMIL reader plus task
classifier. The reported result therefore estimates the value of a frozen
tile geometry and pretrained slide-context initialization under that specific
fine-tuning recipe. It is not a frozen linear-probe estimate of one fixed slide
vector, and it is not end-to-end tile-encoder fine-tuning.

The paper also evaluates on Providence and TCGA tasks and reports patient-level
splitting for the described patient-level mutation setting. External or
temporal cohorts test a different dimension than within-cohort task accuracy.

### 21.5 TITAN

TITAN explicitly compares several learning paradigms:

```text
mean pooling of CONCHv1.5 patches;
ABMIL with patch features;
linear probing of a pretrained slide embedding;
task-specific fine-tuning from pretrained weights;
fine-tuning from random initialization;
few-shot prototypes and few-shot linear probes;
zero-shot text-prompt classification;
linear Cox survival transfer;
slide and cross-modal retrieval;
report generation.
```

These are not interchangeable validation repetitions. They probe, in order,
first moments, supervised weighted moments, linear accessibility, adaptable
initialization, architecture alone, low-sample geometry, language alignment,
proportional-hazards accessibility, archive-relative geometry, and generative
token sufficiency.

The source paper's external rare-cancer retrieval cohort is especially useful
because it tests archive geometry under institution and acquisition shift. Its
small cohort also means uncertainty and case composition remain central to the
interpretation.

## 22. Evaluation Grid That Identifies The Failure Layer

A useful transfer study varies one layer at a time.

### 22.1 Hold Encoder Fixed, Vary The Downstream Reader

```math
\Phi
\quad\text{fixed},
\qquad
\left(
    \mathcal C_{\mathrm{task}},
    \mathcal R_{\mathrm{task}},
    \mathcal H_{\mathrm{task}}
\right)
\in
\left\{
\begin{aligned}
&\text{mean plus linear},\\
&\text{ABMIL},\\
&\text{top-K},\\
&\text{prototype},\\
&\text{graph or transformer context}
\end{aligned}
\right\}.
```

This estimates downstream accessibility while keeping the pretrained
representation fixed. It should be reported as context, readout, and head
rather than assigning the entire pipeline to one operator.

### 22.2 Hold The Downstream Reader Fixed, Vary Encoder

```math
\left(
    \mathcal C_{\mathrm{task}},
    \mathcal R_{\mathrm{task}},
    \mathcal H_{\mathrm{task}}
\right)
\quad\text{fixed},
\qquad
\Phi
\in
\left\{
\Phi_1,\ldots,\Phi_B
\right\}.
```

This is the cleanest encoder comparison, provided patch extraction,
normalization, split, and optimization are also fixed.

### 22.3 Vary Label Budget

```math
K
\in
\{1,2,4,8,16,\ldots\}.
```

Plotting performance against patient-level label count separates immediate
linear accessibility from asymptotic head capacity.

### 22.4 Vary Domain

```math
d
\in
\left\{
\text{internal},
\text{temporal},
\text{external institution},
\text{scanner},
\text{rare morphology}
\right\}.
```

This identifies whether the learned statistic is stable under the deployment
changes that matter.

### 22.5 Vary Metric

```math
\mathcal M
\in
\left\{
\begin{aligned}
&\text{AUROC or balanced accuracy},\\
&\text{AUPRC},\\
&\text{Brier or calibration error},\\
&\text{concordance and survival calibration},\\
&\text{Recall@K},\\
&\text{subgroup worst-case risk}
\end{aligned}
\right\}.
```

No one member certifies the others.

## 23. C/R/G/S Placement For Transfer

Pretraining and transfer each have their own decomposition:

```math
\begin{aligned}
\widetilde H^{\mathrm{pre}}
&=
\mathcal C_{\mathrm{pre}}
\left(
    H^{\mathrm{pre}};
    G_{\mathrm{pre}},
    S_{\mathrm{pre}}
\right),
\\
Z
&=
\mathcal R_{\mathrm{pre}}
\left(
    \widetilde H^{\mathrm{pre}}
\right),
\\
\widetilde Z^{\mathrm{task}}
&=
\mathcal C_{\mathrm{task}}
\left(
    Z;
    G_{\mathrm{task}},
    S_{\mathrm{task}}
\right),
\\
u^{\mathrm{task}}
&=
\mathcal R_{\mathrm{task}}
\left(
    \widetilde Z^{\mathrm{task}}
\right),
\\
\widehat Y
&=
\mathcal H_{\mathrm{task}}
\left(
    u^{\mathrm{task}}
\right).
\end{aligned}
```

The transfer ledger is:

| Component | Pretraining question | Transfer question |
|---|---|---|
| `C` | Which views, patches, regions, slides, or modalities interact? | Can task labels contextualize patches, slides, patients, or prompts? |
| `R` | Which patch, class token, query token, or embedding is exported? | Mean, attention, top-K, prototype, class token, or memory retrieval? |
| `G` | Which augmentation, spatial, hierarchy, cosine, or language geometry is learned? | Which normalization, metric, coordinates, archive, and domain are used? |
| `S` | Contrastive, teacher, masked, caption, report, or knowledge targets? | Labels, censoring, few-shot support, prompts, or no task supervision? |
| `H` | Which pretraining projection, teacher, contrastive, or decoder head defines the objective? | Linear softmax, Cox, similarity decision, calibrated classifier, or text decoder? |

For frozen transfer:

```math
S_{\mathrm{task}}
\not\longrightarrow
\Phi_{\phi}.
```

For fine-tuning:

```math
S_{\mathrm{task}}
\longrightarrow
\Phi_{\phi}
```

through the task gradient. That arrow is the mathematical boundary between
representation evaluation and representation adaptation.

## 24. Minimal Transfer Reporting Rule

A foundation-model transfer result should report:

```text
1. pretrained object:
   patch vector, patch set, slide vector, multitoken slide object, or
   image-text pair;

2. frozen versus trainable components:
   exact encoder stages, projection layers, context blocks, and readout;

3. downstream hypothesis class:
   linear, prototype, top-K, MIL, graph, transformer, Cox, retrieval, or
   decoder;

4. surviving statistic:
   first moment, weighted first moment, order statistic, contextual token,
   metric neighborhood, or generated sequence;

5. geometry:
   feature normalization, distance, coordinates, prompts, archive, and
   tessellation;

6. independent unit and split:
   patient, specimen, slide, patch, site, and duplicate exclusion;

7. label regime:
   full-data, few-shot, zero-shot, weak labels, censoring, or no labels;

8. evaluation target:
   discrimination, calibration, retrieval, survival, generation, or external
   robustness;

9. uncertainty:
   folds, seeds, support-set repetitions, confidence intervals, and case
   counts;

10. intervention audit:
    readout swap, prompt swap, archive swap, coordinate perturbation,
    external cohort, or fine-tuning ablation.
```

Without these fields, "encoder A transfers better than encoder B" is usually
an underspecified pipeline comparison.

## 25. Bottom Line

Foundation-model transfer has four distinct limits:

```math
\boxed{
\text{raw slide}
\xrightarrow{
    \text{representation gap}
}
\text{exported object}
\xrightarrow{
    \text{readout gap}
}
\text{best task rule in class}
\xrightarrow{
    \text{estimation gap}
}
\text{fitted rule}
\xrightarrow{
    \text{shift gap}
}
\text{deployment behavior}.
}
```

Patch encoders such as UNI and Virchow primarily determine local feature
geometry; the slide reader still decides what WSI statistic survives. CONCH
and PLIP add text-constrained directions, but prompt and top-`K` choices still
define the decision. Prov-GigaPath and TITAN move context into pretraining at
the slide level, but their exported tokens, downstream heads, fine-tuning
recipes, retrieval archives, and evaluation domains remain separate parts of
the claim.

The correct transfer question is:

```math
\boxed{
\text{Which task information survives the pretrained map, which part is
accessible to this reader, how much can be estimated from independent
patients, and does the resulting rule survive deployment shift?}
}
```
