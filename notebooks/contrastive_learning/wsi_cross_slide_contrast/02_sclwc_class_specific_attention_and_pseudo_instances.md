# SCL-WC Class-Specific Attention And Pseudo-Instances

Primary anchor:

- Wang et al. "SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised
  Whole-Slide Image Classification." NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf

## Offline Patch Features

SCL-WC pretrains a Swin Transformer with MoCo v3 and uses it as an offline
feature encoder:

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

Slide losses do not update the original image encoder.

## Class-Specific Attention

```math
A_n
\in
\mathbb{R}^{L_n\times C}.
```

For class `c`:

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
},
```

```math
z_n^{(c)}
=
\sum_{l=1}^{L_n}
\alpha_{n,l}^{(c)}
f_{n,l}.
```

## Pseudo-Instance Support

Sort the attention column for the observed slide class:

```math
A_{n,(1)}^{(Y_n)}
\ge
\cdots
\ge
A_{n,(L_n)}^{(Y_n)}.
```

Top-k and bottom-k receive pseudo-labels:

```math
\widetilde y_{n,(r)}
=
\begin{cases}
1,
&r\le k,
\\
0,
&r>L_n-k.
\end{cases}
```

The intended binary cross-entropy is:

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

The paper's displayed Equation 4 prints a minus sign between the two log terms,
while its text calls the loss binary cross-entropy. The expression above is the
mathematically consistent BCE reading.

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

The same attention creates the slide prediction and its own instance targets.

## Support Stability

```math
g_n^{(k)}
=
A_{n,(k)}^{(Y_n)}
-
A_{n,(k+1)}^{(Y_n)}.
```

When this gap is small, tiny score changes switch pseudo-labels and memory-bank
membership.
