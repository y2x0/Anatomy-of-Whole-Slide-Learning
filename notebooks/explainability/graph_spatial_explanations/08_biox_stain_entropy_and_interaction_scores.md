# BioX-CPath Stain Entropy and Interaction Scores

## 1. Within-Stain Entropy

For stain `s`, normalize its selected node scores:

```math
q_{v\mid s}
=\frac{a'_v}{\sum_{u\in V'_s}a'_u},
\qquad v\in V'_s.
```

BioX-CPath reports entropy

```math
H_s
=-\sum_{v\in V'_s}q_{v\mid s}\log q_{v\mid s}.
```

Low entropy means attention is concentrated among few nodes; high entropy means
the stain's selected attention is diffuse. Entropy does not determine whether
the stain is diagnostically helpful.

## 2. Normalization Matters

Raw entropy depends on the number of selected nodes. A normalized comparison is

```math
\widetilde H_s
=\frac{H_s}{\log |V'_s|}
```

when `|V'_s|>1`. Stains with different node counts should not be compared by
raw entropy alone.

## 3. Stain-Stain Interaction

For types `s` and `t`, aggregate cross-stain GAT weights:

```math
I_{st}
=\frac{1}{|P_{st}|}
\sum_{(u,v)\in P_{st}}\beta_{uv}.
```

This is a mean routing interaction over cross-stain pairs. It is symmetric only
if the graph and averaging convention make it symmetric. A large `I_st` does
not imply a positive class effect.

## 4. Intervention Upgrade

To test whether the interaction supports class `c`, compare

```math
\Delta_{st,c}
=F_c(G)-F_c(G\setminus E_{st}),
```

where `E_st` are cross-stain edges. This changes the graph computation and is a
different estimand from `I_st`.

