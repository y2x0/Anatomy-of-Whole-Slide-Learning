# CLAM Top/Bottom-K Constraints

CLAM uses slide labels to create pseudo-instance constraints from attention
extremes.

For class $c$:

```math
a_{ij}^{(c)}
=
\mathrm{softmax}_j s_c(h_{ij}).
```

For a slide with label $Y_i=c$, define:

```math
T_i^{+}(c)
=
\mathrm{TopK}_{j}\ a_{ij}^{(c)},
```

```math
T_i^{-}(c)
=
\mathrm{BottomK}_{j}\ a_{ij}^{(c)}.
```

The pseudo-label rule is:

```math
\widehat Z_{ij}^{(c)}
=
\begin{cases}
1, & j\in T_i^{+}(c),\\
0, & j\in T_i^{-}(c).
\end{cases}
```

Instances outside the selected extremes are unlabeled for the auxiliary loss.

## Auxiliary Loss

Let:

```math
r_c(h)
=
P_\theta(Z^{(c)}=1\mid h).
```

The instance loss is:

```math
\mathcal{L}_{\mathrm{inst}}^{(c)}
=
-
\frac{1}{K}
\sum_{j\in T_i^{+}(c)}
\log r_c(h_{ij})
-
\frac{1}{K}
\sum_{j\in T_i^{-}(c)}
\log(1-r_c(h_{ij})).
```

The total loss is:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{slide}}
+
\lambda
\sum_c
\mathcal{L}_{\mathrm{inst}}^{(c)}.
```

## Hard-EM Interpretation

CLAM's top/bottom selection can be read as a restricted hard posterior:

```math
\widehat Z_i
=
\arg\max_{z\in\mathcal{Q}_K(a_i,Y_i)}
q_\theta(z\mid H_i,Y_i),
```

where $\mathcal{Q}_K$ is the set of assignments that label only the top and
bottom $K$ attention patches.

The selected pseudo-labels then shape the representation:

```math
H_i
\to
H_i'
```

but inference still uses:

```math
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}h_{ij}.
```

## Why It Helps

Bag labels give weak gradients to instance features. The auxiliary loss adds a
separation pressure:

```math
r_c(h_{ij})
\to
1
\quad
j\in T_i^{+}(c),
```

```math
r_c(h_{ij})
\to
0
\quad
j\in T_i^{-}(c).
```

This can make attention more stable when high-attention patches are true
witnesses.

## Why It Fails

If early attention selects the wrong witness:

```math
T_i^{+}(c)
\not\subset
\{j:Z_{ij}^{(c)}=1\},
```

then the auxiliary loss reinforces that error:

```math
r_c(h_{ij})
\uparrow
\quad
\text{for false pseudo-positive }j.
```

This is confirmation bias.

## C/R/G/S Placement

```text
G:
    absent unless coordinates or regions are added externally

C:
    instance classifier and feature geometry are shaped by pseudo-labels

R:
    class-specific attention selects pseudo-label sets and performs readout

S:
    slide label plus attention-derived top/bottom-k pseudo-instance constraints
```

## Dense Summary

CLAM turns:

```math
Y_i
```

into:

```math
(Y_i,\widehat Z_{ij}\text{ for selected }j).
```

The extra labels are useful exactly when attention extremes approximate true
latent instances.
