# SCL-WC Class-Specific Attention And Pseudo-Instances

Primary anchor:

- Wang et al. "SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised
  Whole-Slide Image Classification." NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf

## Offline Patch Features

SCL-WC first pretrains a Swin Transformer with MoCo v3 and then uses it as an
offline feature encoder. For slide `n`:

```math
F_n
=
\left[
f_{n,1}^{\top};
\ldots;
f_{n,L_n}^{\top}
\right]
\in
\mathbb{R}^{L_n\times d}.
```

The weakly supervised aggregation stage does not update the original image
encoder through slide losses.

## Class-Specific Attention

Let attention matrix be:

```math
A_n
\in
\mathbb{R}^{L_n\times C}.
```

Column `c` assigns a class-specific score to every patch. Normalized weights
for class `c` are:

```math
\alpha_{n,l}^{(c)}
=
\frac{
\exp
\left(
A_{n,l}^{(c)}
\right)
}{
\sum_{r=1}^{L_n}
\exp
\left(
A_{n,r}^{(c)}
\right)
}.
```

The class-conditioned slide feature is:

```math
z_n^{(c)}
=
\sum_{l=1}^{L_n}
\alpha_{n,l}^{(c)}
f_{n,l}.
```

## Pseudo-Instance Support

For the observed class `Y_n`, sort:

```math
A_{n,(1)}^{(Y_n)}
\ge
\cdots
\ge
A_{n,(L_n)}^{(Y_n)}.
```

Top-k patches receive pseudo-label one and bottom-k patches pseudo-label zero:

```math
\widetilde y_{n,(r)}
=
\begin{cases}
1,
&
r\le k,
\\
0,
&
r>L_n-k.
\end{cases}
```

The support is attention-defined rather than pathologist-observed.

## Instance Loss

For predicted instance probabilities `p_j`, the intended binary
cross-entropy form is:

```math
\mathcal{L}_{\mathrm{ins}}
=
-\frac{1}{2k}
\sum_{j=1}^{2k}
\left[
\widetilde y_j\log p_j
+
\left(
1-\widetilde y_j
\right)
\log
\left(
1-p_j
\right)
\right].
```

The paper's displayed Equation 4 prints a minus sign between the two log terms;
the surrounding description calls it binary cross-entropy. The standard BCE
above is the mathematically consistent reading.

## CDA Objective

```math
\mathcal{L}_{\mathrm{CDA}}
=
\lambda_1
\mathcal{L}_{\mathrm{MIL}}
+
\lambda_2
\mathcal{L}_{\mathrm{ins}}.
```

The same attention values affect slide prediction and generate instance
targets, creating a self-reinforcing support-selection path.

## Support Stability

Define the top-k boundary gap:

```math
g_n^{(k)}
=
A_{n,(k)}^{(Y_n)}
-
A_{n,(k+1)}^{(Y_n)}.
```

Small `g_n^(k)` means tiny score perturbations can switch pseudo-labels and
memory-bank membership.
