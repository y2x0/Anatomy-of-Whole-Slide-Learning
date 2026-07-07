# CLAM Instance Clustering

CLAM adds an auxiliary instance-level constraint to class-specific attention.
The key idea is not that true patch labels are known. They are not. Instead,
attention ranks patches, and the model treats attention extremes as pseudo
instances for representation shaping.

## Slide-Level Attention

For class $c$:

```math
a_{ij}^{(c)}
=
\mathrm{softmax}_{j}s_c(h_{ij}),
\qquad
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}h_{ij}.
```

The slide logit is:

```math
o_i^{(c)}
=
g_c(z_i^{(c)}).
```

Slide supervision gives:

```math
\mathcal{L}_{\text{slide}}
=
\mathrm{CE}(y_i,\widehat y_i).
```

## Attention Extremes

For the true slide class $y_i=c$, define the top-attention set:

```math
T_i^{+}(c)
=
\mathrm{TopK}_{j}\ a_{ij}^{(c)}.
```

Define the low-attention set:

```math
T_i^{-}(c)
=
\mathrm{BottomK}_{j}\ a_{ij}^{(c)}.
```

CLAM uses these sets as pseudo positive and pseudo negative examples for the
class branch.

## Instance Classifier

Let an instance classifier for class $c$ be:

```math
r_c(h)
=
\mathrm{MLP}_c(h).
```

The auxiliary instance loss can be written:

```math
\mathcal{L}_{\text{inst}}^{(c)}
=
\frac{1}{K}
\sum_{j\in T_i^{+}(c)}
\mathrm{CE}(r_c(h_{ij}),1)
+
\frac{1}{K}
\sum_{j\in T_i^{-}(c)}
\mathrm{CE}(r_c(h_{ij}),0).
```

For classes absent from the slide, high-attention candidates for those branches
can be treated as negative evidence. The exact bookkeeping differs by single-
branch versus multi-branch variants, but the mathematical role is the same:
shape embeddings so high-attention patches form class-separable clusters.

## Total Objective

The training objective is:

```math
\mathcal{L}
=
\mathcal{L}_{\text{slide}}
+
\lambda\mathcal{L}_{\text{inst}}.
```

The auxiliary term changes the instance representation even though labels remain
slide-level. It says:

```text
patches selected as high attention for the slide class should look separable
from low-attention patches
```

## What Statistic Survives?

At inference, the readout is still:

```math
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}h_{ij}.
```

The clustering loss does not add a new readout statistic. It changes the feature
geometry from which the same class-specific weighted first moment is computed.

## Dense Summary

CLAM can be decomposed as:

```math
H_i
\xrightarrow{\text{class attention}}
a_i^{(c)}
\xrightarrow{\text{top/bottom selection}}
T_i^{+}(c),T_i^{-}(c)
\xrightarrow{\text{instance loss}}
\text{feature shaping}
\xrightarrow{\text{weighted sum}}
z_i^{(c)}.
```

The slide prediction still sees only a class-conditioned weighted average, but
training pressures the embedding space to make attention extremes more
class-separable.

