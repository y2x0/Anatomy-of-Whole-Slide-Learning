# CLAM Class-Specific Attention and Clustering

## 1. Scope and source

CLAM changes attention MIL in two coupled ways:

1. the slide representation is class-specific, so different target branches can
   rank the same patch differently;
2. an instance-level auxiliary objective turns attention extremes into
   pseudo-instance supervision.

The primary source is Lu et al., Data-Efficient and Weakly Supervised
Computational Pathology on Whole-Slide Images:

https://arxiv.org/abs/2004.09666

The important phrase is pseudo-instance supervision. CLAM does not observe
patch-level tumor labels in the weakly supervised setting. It selects top and
bottom attention instances using the current model and trains an instance
classifier on those generated targets.

This creates a useful localization pressure and a feedback loop:

```math
\text{bag label}
\longrightarrow
\text{attention ranking}
\longrightarrow
\text{pseudo-instance targets}
\longrightarrow
\text{attention and feature updates}.
```

The loop is the mathematical source of both CLAM's interpretability benefit
and its confirmation-bias failure.

## 2. Input and branch notation

Let slide i contain n_i patch embeddings of width d_0:

```math
X_i
=
\begin{bmatrix}
x_{i1}^{\mathsf T}\\
\vdots\\
x_{in_i}^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{n_i\times d_0}.
```

CLAM first maps patches into an attention and classifier width d:

```math
h_{ij}
=
\mathrm{ReLU}(W_{\mathrm{in}}x_{ij}+b_{\mathrm{in}})
\in\mathbb R^d.
```

For C target classes, CLAM-MB uses one attention branch per class. Let the
attention hidden width be r. The shared gated feature is

```math
q_{ij}
=
\tanh(Vh_{ij})
\odot
\mathrm{sigm}(Uh_{ij}),
```

with

```math
V,U\in\mathbb R^{r\times d}.
```

The class-specific unnormalized score is

```math
e_{ij}^{(c)}
=
\left(w_a^{(c)}\right)^{\mathsf T}q_{ij},
\qquad
w_a^{(c)}\in\mathbb R^r.
```

The branch attention vector is normalized over patches:

```math
\alpha_{ij}^{(c)}
=
\frac{\exp(e_{ij}^{(c)})}
{\sum_{\ell=1}^{n_i}\exp(e_{i\ell}^{(c)})}.
```

The class-specific slide representation is

```math
z_i^{(c)}
=
\sum_{j=1}^{n_i}\alpha_{ij}^{(c)}h_{ij}
\in\mathbb R^d.
```

## 3. CLAM-MB and CLAM-SB

### 3.1 Multi-branch model

CLAM-MB applies one scalar classifier to each class representation:

```math
\ell_i^{(c)}
=
\left(w_{\mathrm{cls}}^{(c)}\right)^{\mathsf T}z_i^{(c)}
+
b_{\mathrm{cls}}^{(c)}.
```

The bag probability is

```math
\widehat p_i^{(c)}
=
\mathrm{softmax}\!\left(
\ell_i^{(1)},\ldots,\ell_i^{(C)}
\right)_c.
```

The branch index enters both attention and readout.

### 3.2 Single-branch model

CLAM-SB uses one attention vector:

```math
\alpha_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{\ell}\exp(e_{i\ell})},
\qquad
z_i
=
\sum_j\alpha_{ij}h_{ij}.
```

The classifier then produces all class logits:

```math
\ell_i
=
W_{\mathrm{cls}}z_i+b_{\mathrm{cls}}.
```

CLAM-SB can provide one shared attention map for a multiclass prediction. It
does not have the branch-specific rankings of CLAM-MB.

The distinction is:

```math
\mathrm{CLAM\text{-}MB}:
\quad
\{\alpha_{ij}^{(c)}\}_{c=1}^{C},
\qquad
\mathrm{CLAM\text{-}SB}:
\quad
\{\alpha_{ij}\}.
```

Calling a CLAM-MB heatmap the slide's single intrinsic importance map discards
the class index.

## 4. Permutation invariance

Let P_i permute patch rows. The gated feature is rowwise:

```math
q(P_iH_i)
=
P_iq(H_i).
```

Each branch score permutes in the same way:

```math
e^{(c)}(P_iH_i)
=
P_ie^{(c)}(H_i).
```

Softmax and the weighted sum preserve the branch representation:

```math
z_i^{(c)}(P_iH_i)
=
z_i^{(c)}(H_i).
```

Therefore CLAM attention is set-invariant over the supplied patch-feature
pairs. It has no pairwise spatial context unless coordinates or local context
have already entered h_ij.

The top-k index set permutes rather than remaining numerically fixed:

```math
I_{i,+}^{(c)}(P_iH_i)
=
P_iI_{i,+}^{(c)}(H_i).
```

The selected features and pseudo-label objective are therefore invariant as a
set operation, apart from exact ties and implementation tie-breaking.

## 5. Class-conditional attention geometry

For two classes c and c prime, the score difference is

```math
\Delta e_{ij}^{(c,c')}
=
\left(
w_a^{(c)}-w_a^{(c')}
\right)^{\mathsf T}q_{ij}.
```

The ratio of branch weights is not simply the exponential of this difference
because each branch has its own normalization constant:

```math
\frac{\alpha_{ij}^{(c)}}{\alpha_{ij}^{(c')}}
=
\exp\!\left(
\Delta e_{ij}^{(c,c')}
\right)
\frac{
\sum_\ell\exp(e_{i\ell}^{(c')})
}{
\sum_\ell\exp(e_{i\ell}^{(c)})
}.
```

One patch can therefore have high weight for two classes if each branch ranks
it highly relative to a different bag-wide score distribution.

A class-contrast routing score can be defined as

```math
\Delta\alpha_{ij}^{(c,c')}
=
\alpha_{ij}^{(c)}-\alpha_{ij}^{(c')}.
```

This measures relative routing, not relative class evidence. The class head
still acts on z_i^(c) and z_i^(c prime).

## 6. Bag loss

For observed slide class y_i, the bag loss is standard multiclass cross
entropy:

```math
\mathcal L_{\mathrm{bag},i}
=
-\log\widehat p_i^{(y_i)}.
```

For binary logits, the equivalent logistic form is

```math
\mathcal L_{\mathrm{bag},i}
=
-y_i\log\sigma(\ell_i)
-
(1-y_i)\log(1-\sigma(\ell_i)).
```

The bag label is the observed supervision channel. The patch labels used by
the auxiliary objective are generated from attention and are not observations.

## 7. Top and bottom selection

Let k be the instance sample count, with k less than or equal to n_i. For
branch c, define the top and bottom index sets:

```math
I_{i,+}^{(c)}
=
\mathrm{TopK}_k
\left(
\{\alpha_{ij}^{(c)}\}_{j=1}^{n_i}
\right),
```

```math
I_{i,-}^{(c)}
=
\mathrm{BottomK}_k
\left(
\{\alpha_{ij}^{(c)}\}_{j=1}^{n_i}
\right).
```

Equivalently, top selection uses the k largest scores and bottom selection uses
the k smallest scores because softmax is strictly increasing:

```math
\mathrm{TopK}_k(\alpha^{(c)})
=
\mathrm{TopK}_k(e^{(c)}).
```

The pseudo-label assignment is a rule, not a posterior:

```math
\widetilde y_{ij}^{(c)}
=
\begin{cases}
1,&j\in I_{i,+}^{(c)},\\
0,&j\in I_{i,-}^{(c)}.
\end{cases}
```

The remaining n_i minus 2k instances receive no instance-level target in the
in-class branch.

## 8. In-class instance evaluation

For the branch corresponding to the observed class y_i, the selected feature
set is

```math
\widetilde H_{i,\mathrm{in}}^{(y_i)}
=
\{h_{ij}:j\in I_{i,+}^{(y_i)}\}
\cup
\{h_{ij}:j\in I_{i,-}^{(y_i)}\}.
```

The targets are k ones followed by k zeros:

```math
\widetilde Y_{i,\mathrm{in}}^{(y_i)}
=
\begin{bmatrix}
\mathbf 1_k\\
\mathbf 0_k
\end{bmatrix}.
```

Let the branch instance classifier be

```math
r_{ij}^{(c)}
=
W_{\mathrm{inst}}^{(c)}h_{ij}
+
b_{\mathrm{inst}}^{(c)}
\in\mathbb R^2.
```

The in-class loss is

```math
\mathcal L_{\mathrm{inst},i}^{(y_i)}
=
\frac{1}{2k}
\sum_{j\in I_{i,+}^{(y_i)}}
\mathrm{CE}\!\left(r_{ij}^{(y_i)},1\right)
+
\frac{1}{2k}
\sum_{j\in I_{i,-}^{(y_i)}}
\mathrm{CE}\!\left(r_{ij}^{(y_i)},0\right).
```

The phrase clustering loss refers to this pressure to separate selected
features in the instance classifier, not to an unsupervised k-means objective.

## 9. Out-of-class instance evaluation

For a branch c not equal to the observed class y_i, CLAM's out-of-class rule
can label the branch's top attention instances as negative:

```math
\widetilde y_{ij}^{(c)}
=
0
\qquad
j\in I_{i,+}^{(c)},
\quad
c\ne y_i.
```

The corresponding loss is

```math
\mathcal L_{\mathrm{inst},i}^{(c,\mathrm{out})}
=
\frac{1}{k}
\sum_{j\in I_{i,+}^{(c)}}
\mathrm{CE}\!\left(r_{ij}^{(c)},0\right).
```

This rule is a mutual-exclusivity assumption: a patch selected as highly
relevant to an out-of-class branch is treated as negative for that branch in a
slide belonging to another class.

It is not valid without modification for multilabel pathology where multiple
morphologies can legitimately coexist in one slide.

## 10. Total objective

Let the instance loss average over the relevant class branches. The total CLAM
objective has the form

```math
\mathcal L_i
=
\mathcal L_{\mathrm{bag},i}
+
\lambda_{\mathrm{inst}}
\mathcal L_{\mathrm{inst},i}.
```

For a multi-branch implementation:

```math
\mathcal L_{\mathrm{inst},i}
=
\frac{1}{C}
\left[
\mathcal L_{\mathrm{inst},i}^{(y_i)}
+
\sum_{c\ne y_i}
\mathcal L_{\mathrm{inst},i}^{(c,\mathrm{out})}
\right].
```

The exact reduction and branch handling are implementation choices, but the
structural decomposition is stable:

```math
\text{observed slide loss}
+
\text{attention-selected pseudo-instance loss}.
```

The auxiliary term can sharpen instance features even though its targets are
model-generated.

## 11. The pseudo-label feedback loop

At training step t, write the attention scores as e_t. Selection is

```math
\widetilde Y_t
=
\mathcal S_k(e_t,y_i),
```

where S includes top and bottom rules. The next parameter update is

```math
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta
\left[
\mathcal L_{\mathrm{bag}}(\theta_t)
+
\lambda_{\mathrm{inst}}
\mathcal L_{\mathrm{inst}}(\theta_t;\widetilde Y_t)
\right].
```

The pseudo-label is treated as fixed inside the selected update. Hard top-k has
zero derivative with respect to the selected indices away from rank boundaries:

```math
\frac{\partial\widetilde Y_t}{\partial e_t}
=
0
\quad\text{locally away from ties}.
```

The instance loss still updates the selected feature encoder and instance
classifier. The attention scorer receives direct pressure primarily through the
bag loss and shared upstream parameters, not through a smooth derivative of the
top-k operation.

At a rank swap, the selected set changes discontinuously:

```math
e_{ij}=e_{i\ell}
\quad\Longrightarrow\quad
\mathcal S_k(e)
\text{ can change under an arbitrarily small perturbation}.
```

This is a source of optimization noise and explanation instability.

## 12. Pseudo-label quality

Let T_i^(c) be the unknown true set of positive instances for class c. The
precision of the generated top set is

```math
\mathrm{Prec}_{i,+}^{(c)}
=
\frac{
|I_{i,+}^{(c)}\cap T_i^{(c)}|
}{
|I_{i,+}^{(c)}|
}.
```

The bottom-set negative purity is

```math
\mathrm{Purity}_{i,-}^{(c)}
=
\frac{
|I_{i,-}^{(c)}\cap (T_i^{(c)})^c|
}{
|I_{i,-}^{(c)}|
}.
```

These quantities are not observed under weak supervision. A high bag
validation score does not prove high pseudo-label purity.

If top precision is p and bottom purity is q, the expected label noise in the
selected 2k examples is approximately

```math
\eta_{\mathrm{sel}}
=
\frac{1}{2}\left[(1-p)+(1-q)\right].
```

This noise is not independent random noise. It is correlated with the current
attention model and can reinforce its errors.

## 13. Confirmation bias

Suppose an artifact feature a receives high attention because it correlates
with the slide label in the training cohort. Then

```math
a\in I_{i,+}^{(y_i)}
\quad\Longrightarrow\quad
\widetilde y_{ia}=1.
```

The instance loss trains the branch classifier to recognize a as positive. The
updated representation can increase the artifact score, making the next top-k
selection more likely to include a:

```math
\mathrm{score}_{t+1}(a)
>
\mathrm{score}_t(a)
\quad\text{under the shortcut update}.
```

This is self-training on a label generated by the same model. It can sharpen a
wrong heatmap rather than correct it.

The failure is especially severe when:

- the true positive region is rare;
- the artifact is spatially repeated;
- the class is imbalanced;
- the top-k set is small;
- external validation shifts the acquisition shortcut.

## 14. Class-specific heatmaps

For predicted class c-hat, the standard displayed heatmap may use

```math
\left\{
\alpha_{ij}^{(\widehat c)}
\right\}_{j=1}^{n_i}.
```

This answers:

```math
\text{which patches received the largest normalized routing mass in branch
\widehat c?}
```

It does not directly answer:

```math
\text{which patches have the largest signed contribution to the class margin?}
```

For a linear class head, a branch-level margin contribution can be approximated
by

```math
\gamma_{ij}^{(c,c')}
=
\alpha_{ij}^{(c)}
\left(w_{\mathrm{cls}}^{(c)}\right)^{\mathsf T}h_{ij}
-
\alpha_{ij}^{(c')}
\left(w_{\mathrm{cls}}^{(c')}\right)^{\mathsf T}h_{ij}.
```

Even this expression omits nonlinear head effects and interactions. A heatmap
must name its branch and score definition.

## 15. Attention versus instance classifier

The instance classifier output is

```math
\widehat y_{ij}^{\mathrm{inst},(c)}
=
\mathrm{softmax}\!\left(r_{ij}^{(c)}\right)_1.
```

The attention score and instance probability are trained by different paths:

```math
\alpha_{ij}^{(c)}
=
\mathrm{softmax}\!\left(e_{ij}^{(c)}\right)_j,
\qquad
\widehat y_{ij}^{\mathrm{inst},(c)}
=
\mathrm{softmax}\!\left(r_{ij}^{(c)}\right)_1.
```

They need not rank patches identically after training. Agreement is evidence
that the auxiliary classifier and attention branch have aligned, not an
identity imposed by the equations.

The instance classifier can also be evaluated on all patches after training,
while the training target existed only for selected extremes. Its output on
unselected patches is extrapolation from a partially labeled training set.

## 16. Top-k as a readout and supervision operator

CLAM uses attention in two roles:

```math
\text{attention}
\longrightarrow
\begin{cases}
\text{weighted slide representation},\\
\text{selected pseudo-instance set}.
\end{cases}
```

The first role is differentiable through softmax. The second role is hard and
rank-based. Increasing k changes both:

- the amount of evidence in the instance loss;
- the label-noise exposure;
- the selection stability;
- the computational cost of the auxiliary classifier.

If k is too small, one shortcut patch can dominate. If k is too large, the
selected top set can include background and the bottom set can include
unrecognized positives.

## 17. Bag-size dependence of top-k

The top-k fraction is

```math
\rho_{i,k}
=
\frac{k}{n_i}.
```

Fixed k means that larger slides receive a smaller supervised fraction:

```math
n_i'>n_i
\quad\Longrightarrow\quad
\rho_{i',k}<\rho_{i,k}.
```

Fixed fraction means variable k and different auxiliary loss size. The choice
should be reported because it changes the pseudo-label coverage.

The expected chance that a uniformly distributed positive patch enters a
random k-set is k over n_i. CLAM is not random, but this baseline shows how
quickly fixed-k coverage vanishes on large bags.

## 18. Mutual exclusivity and multilabel pathology

For a single-label classification problem, an out-of-class branch can use the
observed class to define a negative constraint. For multilabel targets y_i in
\{0,1\}^C, a branch c should instead use

```math
\widetilde y_{ij}^{(c)}
=
\begin{cases}
1,&y_i^{(c)}=1\text{ and }j\in I_{i,+}^{(c)},\\
0,&y_i^{(c)}=0\text{ and }j\in I_{i,+}^{(c)},\\
\text{unknown},&\text{otherwise}.
\end{cases}
```

Treating every non-observed class as negative creates false negative
instance constraints when several pathologies coexist.

## 19. Geometry

CLAM's bag-level attention is a set operation. If patch order changes, the
class-specific map permutes and the slide representation is unchanged:

```math
z_i^{(c)}(P_iH_i)=z_i^{(c)}(H_i).
```

The rendered heatmap uses coordinates after prediction:

```math
E_i^{(c)}(x)
=
\sum_j\alpha_{ij}^{(c)}
\mathbf 1\{x\in R_{ij}\},
```

where R_ij is the physical patch region. Rendering adds spatial display but
does not make the model spatially contextual.

Two slides with the same patch-feature multiset and different arrangements
receive the same branch representation. The heatmap can look different only
because the same scores are placed at different coordinates.

## 20. Failure modes

### 20.1 Shortcut confirmation

Early high attention to a scanner or tissue artifact generates its own
pseudo-positive target and reinforces the shortcut.

### 20.2 Positive-bag contamination

A positive slide can contain many negative patches. Top-k selection may improve
precision, but bottom-k is not guaranteed to be a true negative set when
positive morphology is heterogeneous or attention is initially wrong.

### 20.3 Negative-bag contamination

A negative slide can contain morphologically suspicious but non-diagnostic
patches. Forcing top attention patches to negative can train the instance
classifier to suppress a legitimate precursor or confounder.

### 20.4 Rank instability

Near-tied attention scores can change top-k membership under small perturbations:

```math
|e_{ij}-e_{i\ell}|\approx 0
\quad\Longrightarrow\quad
\mathrm{TopK}_k(e)
\text{ is unstable}.
```

### 20.5 Branch inconsistency

Different class branches may focus on overlapping or contradictory regions.
This is not necessarily a bug, but a single merged heatmap can hide the target
dependence.

### 20.6 Patch-count shortcut

Softmax normalization and fixed k make the total supervised fraction depend on
bag size. The model can exploit tissue fraction or count.

### 20.7 No geometry

CLAM can localize high-scoring patches but cannot learn that two patches are
adjacent, separated, or part of a gland unless that relation is encoded before
the bag operator.

## 21. Sanity checks

### Check A: top-k stability

Perturb features or scores slightly and measure Jaccard overlap:

```math
\mathrm{Jaccard}_{i}^{(c)}
=
\frac{
|I_{i,+}^{(c)}\cap I_{i,+}^{\prime(c)}|
}{
|I_{i,+}^{(c)}\cup I_{i,+}^{\prime(c)}|
}.
```

Low overlap indicates a fragile pseudo-label channel.

### Check B: attention versus deletion

Rank patches by branch attention and by class-logit deletion. Report their
correlation rather than assuming that top attention is top evidence.

### Check C: pseudo-label precision

On a small annotated subset, estimate top and bottom purity. This is the direct
test of the auxiliary target assumption.

### Check D: k sweep

Vary k while holding all other settings fixed. Report bag performance,
instance purity, heatmap stability, and effective support.

### Check E: branch contrast

Compare class-specific maps and margin scores:

```math
\alpha^{(c)}-\alpha^{(c')},
\qquad
\gamma^{(c,c')}.
```

This distinguishes routing differences from signed target evidence.

### Check F: multilabel stress

Construct slides containing two valid morphologies and verify that an
out-of-class negative rule does not suppress the second morphology.

## 22. C/R/G/S placement

| Component | CLAM realization |
|---|---|
| Context operator | Pointwise gated attention score; no patch-patch context at the bag level. |
| Readout operator | Class-specific softmax-weighted first moment in CLAM-MB, or shared attention in CLAM-SB. |
| Geometry | Set-invariant over patch-feature pairs; coordinates are used for rendering unless encoded upstream. |
| Surviving statistic | A class-conditional weighted mean plus an attention-derived top/bottom pseudo-instance set. |

The full construction is

```math
\text{patch features}
\xrightarrow{\text{class branches}}
\text{attention scores}
\xrightarrow{\text{softmax}}
\text{class-specific moments}
\xrightarrow{\text{bag head}}
\text{slide logits},
```

with a second supervision path:

```math
\text{attention extremes}
\xrightarrow{\text{hard TopK/BottomK}}
\text{pseudo-instance labels}
\xrightarrow{\text{instance CE}}
\text{feature and classifier updates}.
```

The C/R/G/S distinction exposes that the clustering loss is a supervision
operator, not part of the class-specific readout itself.

## 23. Bottom line

CLAM is class-specific attention MIL plus a self-generated instance-training
loop. The branch representation is a class-conditional weighted first moment.
The auxiliary objective selects attention extremes and labels them using the
bag-level class context.

This can make heatmaps sharper and improve feature separation. It does not
convert weak slide labels into ground-truth patch labels. The generated
pseudo-labels are conditional on the current attention model, hard top-k
selection is rank-discontinuous, and out-of-class constraints assume
mutual exclusivity.

The honest summary is:

```math
\boxed{
\text{CLAM}
=
\text{class-specific weighted moment}
\;+\;
\text{attention-derived pseudo-instance constraints}
\;+\;
\text{confirmation-bias risk}
}
```

A CLAM heatmap should name the target branch, the selection rule, and the
validation test used to support localization. It is a class-conditional
routing map with generated supervision, not an intrinsic tumor probability.

## References

- Lu et al., "Data-efficient and weakly supervised computational pathology on
  whole-slide images," Nature Biomedical Engineering 2021.
  https://arxiv.org/abs/2004.09666
- Ilse et al., "Attention-based Deep Multiple Instance Learning,"
  https://arxiv.org/abs/1802.04712
- Jain and Wallace, "Attention is not Explanation,"
  https://arxiv.org/abs/1902.10186
- Zaheer et al., "Deep Sets," https://arxiv.org/abs/1703.06114
