# CLAM Top/Bottom-K Constraints

CLAM does not merely add a generic instance cross-entropy loss. Its auxiliary
signal is a clustering constraint built from attention extremes. The observed
signal is still the slide label:

```math
S_i^{\mathrm{obs}}
=
Y_i.
```

The generated target is a set of high-attention and low-attention patches:

```math
(\mathcal{T}_i^+(c),\mathcal{T}_i^-(c))
=
\Psi_{\mathrm{CLAM}}(A_i,Y_i).
```

CLAM is therefore a slide-label method plus an attention-derived instance
constraint, not an instance-labeled method.

Reference: [CLAM paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8711640/).

## Attention Extremes

For class
```math
c
```
, let the class-specific attention score be:

```math
a_{ij}^{(c)}
=
\frac{\exp s_c(h_{ij})}
{\sum_{\ell=1}^{n_i}\exp s_c(h_{i\ell})}.
```

For a slide whose observed class is
```math
Y_i=c
```
, CLAM selects:

```math
\mathcal{T}_i^+(c)
=
\mathrm{TopK}_{j}\ a_{ij}^{(c)},
```

```math
\mathcal{T}_i^-(c)
=
\mathrm{BottomK}_{j}\ a_{ij}^{(c)}.
```

The implicit pseudo-labels are:

```math
\widehat Z_{ij}^{(c)}
=
1
\quad
j\in\mathcal{T}_i^+(c),
```

```math
\widehat Z_{ij}^{(c)}
=
0
\quad
j\in\mathcal{T}_i^-(c).
```

Instances outside the selected extremes do not receive an auxiliary instance
target.

## Smooth SVM Clustering Loss

The CLAM auxiliary loss is usually described as instance-level clustering with
a binary top-1 smooth SVM loss, not ordinary instance cross-entropy.

Let the instance classifier for class branch
```math
c
```
produce two scores:

```math
r_c(h)
=
(r_{c0}(h),r_{c1}(h)).
```

For binary pseudo-label
```math
y\in\{0,1\}
```
, the top-1 smooth SVM surrogate can be
written schematically as:

```math
\ell_{\mathrm{svm}}(r_c(h),y)
=
\tau
\log
\sum_{q\in\{0,1\}}
\exp
\left(
\frac{
\alpha\mathbf{1}\{q\ne y\}
+r_{cq}(h)-r_{cy}(h)
}{\tau}
\right).
```

Here
```math
\alpha
```
 is the margin parameter and
```math
\tau
```
smooths the hard maximum. The
selected-extreme loss for the true class branch is:

```math
\mathcal{L}_{\mathrm{inst}}^{\mathrm{in}}(i,c)
=
\frac{1}{2K}
\left[
\sum_{j\in\mathcal{T}_i^+(c)}
\ell_{\mathrm{svm}}(r_c(h_{ij}),1)
+
\sum_{j\in\mathcal{T}_i^-(c)}
\ell_{\mathrm{svm}}(r_c(h_{ij}),0)
\right].
```

The total training objective combines slide classification and the auxiliary
clustering loss:

```math
\mathcal{L}_{\mathrm{CLAM}}
=
\mathcal{L}_{\mathrm{slide}}
+
\lambda
\mathcal{L}_{\mathrm{inst}}.
```

## CLAM-SB Versus CLAM-MB

CLAM-SB uses a single attention branch for the slide representation:

```math
z_i
=
\sum_j a_{ij}v(h_{ij}),
```

followed by a multi-class slide classifier.

CLAM-MB uses class-specific attention branches:

```math
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}v_c(h_{ij}),
```

and class-specific slide evidence:

```math
o_{ic}
=
g_c(z_i^{(c)}).
```

For subtype classification, CLAM-MB also uses a mutual-exclusivity assumption:
high-attention patches for an absent class can be treated as negative examples
for that class. For
```math
c\ne Y_i
```
:

```math
\mathcal{T}_i^{\mathrm{out}}(c)
=
\mathrm{TopK}_{j}\ a_{ij}^{(c)},
```

```math
\mathcal{L}_{\mathrm{inst}}^{\mathrm{out}}(i,c)
=
\frac{1}{K}
\sum_{j\in\mathcal{T}_i^{\mathrm{out}}(c)}
\ell_{\mathrm{svm}}(r_c(h_{ij}),0).
```

This is not a universally valid biological statement. It depends on the task
being mutually exclusive at the slide level and on class-specific attention not
locking onto shared morphology.

## Why It Helps

The slide loss only requires:

```math
g\left(\sum_j a_{ij}^{(c)}v_c(h_{ij})\right)
\approx
Y_i.
```

It does not require attention to identify true diagnostic patches. The
clustering loss adds a stronger inductive bias:

```math
j\in\mathcal{T}_i^+(c)
\Rightarrow
r_{c1}(h_{ij})\uparrow,
```

```math
j\in\mathcal{T}_i^-(c)
\Rightarrow
r_{c1}(h_{ij})\downarrow.
```

The mathematical bet is:

```math
P(Z_{ij}^{(c)}=1\mid j\in\mathcal{T}_i^+(c),Y_i=c)
>
P(Z_{ij}^{(c)}=1\mid Y_i=c).
```

If this enrichment condition is true, the auxiliary loss can stabilize the
instance geometry.

## Why It Fails

If attention selects a shortcut witness:

```math
\mathcal{T}_i^+(c)
\cap
\{j:Z_{ij}^{(c)}=1\}
=
\varnothing,
```

then the auxiliary loss explicitly teaches the instance classifier that the
shortcut is positive:

```math
r_{c1}(h_{ij})
\uparrow
\quad
j\in\mathcal{T}_i^+(c).
```

The top/bottom rule is therefore a model-generated MNAR selection mechanism:

```math
P(j\in\mathcal{T}_i^+(c)\mid Z_{ij},H_i,\theta_t)
\ne
P(j\in\mathcal{T}_i^+(c)).
```

## Not A Literal EM Algorithm

CLAM can be described as hard-assignment-like:

```math
\widehat Z_i^{\mathrm{sel}}
=
\Psi_{\mathrm{CLAM}}(A_i,Y_i).
```

But unless one specifies a likelihood for
```math
Y_i
```
,
```math
A_i
```
, and
```math
Z_i
```
such that
top/bottom-k selection is the exact MAP posterior step, CLAM is not literally
hard EM. The safer statement is:

```text
CLAM uses attention-selected pseudo-instance constraints.
```

## C/R/G/S Placement

```text
G:
    absent unless coordinates or regions are added externally

C:
    instance classifier and feature geometry are shaped by smooth-SVM clustering

R:
    attention both reads out the slide and selects pseudo-instance extremes

S:
    observed slide label plus attention-generated top/bottom-k constraints
```

## Dense Summary

CLAM turns:

```math
Y_i
```

into:

```math
\left(
Y_i,
\mathcal{T}_i^+(Y_i),
\mathcal{T}_i^-(Y_i)
\right).
```

The auxiliary target is useful only when attention extremes are enriched for
true class evidence. The method's strength and its failure mode are the same:
it trusts its own attention.
