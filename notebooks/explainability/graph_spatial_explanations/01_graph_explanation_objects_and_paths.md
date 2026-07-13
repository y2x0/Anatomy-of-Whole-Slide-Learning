# Graph Explanation Objects and Paths

## 1. Graph Forward Map

For a WSI graph

```math
G_i=(V_i,E_i,X_i,A_i),
```

write a message-passing network as

```math
H_i^{(0)}=X_i,
\qquad
H_i^{(\ell+1)}
=\mathcal C_\ell(H_i^{(\ell)},A_i,E_i),
```

and a graph readout and head as

```math
z_i=\mathcal R(H_i^{(L)},A_i,E_i),
\qquad
F_i=\mathcal H(z_i).
```

An explanation can target any argument of this composition.

## 2. Five Non-equivalent Scores

```math
\begin{aligned}
\text{node attention:}&\quad a_v,\\
\text{node removal effect:}&\quad
F(G)-F(G\setminus v),\\
\text{edge removal effect:}&\quad
F(G)-F(G\setminus e),\\
\text{path contribution:}&\quad
\partial F/\partial m_{u\to v},\\
\text{topology effect:}&\quad
F(G(A))-F(G(A')).
\end{aligned}
```

The first is a learned routing quantity, the next two are model perturbations,
the fourth is a local differential quantity, and the fifth compares graph
construction hypotheses.

## 3. Spatial Meaning Is External

If node coordinates are `c_v`, a visualization map is

```math
\mathcal M(c_v)=s_v.
```

This becomes a spatial explanation only when `v` corresponds to a spatially
defined region and the score `s_v` has an explicitly stated target. A graph
constructed by feature-space k-nearest neighbors may be drawn on the slide, but
its edges do not thereby become physical adjacency.

## 4. C/R/G/S Placement

```text
C: message passing and attention move information along E
R: node, type, stain, or graph pooling compresses contextual states
G: coordinates, region adjacency, feature similarity, and topology define E
S: slide labels, pseudo-types, stain labels, or localization hypotheses
```

The explanation must name which of these four locations it evaluates.

