# Cell Graph And Tissue Graph Construction

HACT builds two graphs from a histology region:

```math
I_i
\longmapsto
\left(
G_i^{\mathrm{cell}},
G_i^{\mathrm{tissue}}
\right).
```

The two graphs have different node types, different edge semantics, and
different feature construction procedures.

## Cell Graph

The cell graph is:

```math
G_i^{\mathrm{cell}}
=
\left(
V_i^{\mathrm{cell}},
E_i^{\mathrm{cell}},
H_i^{\mathrm{cell}}
\right).
```

Each node is a detected nucleus or cell entity.

Let the detected cell centroids be:

```math
c_{ia}^{\mathrm{cell}}
\in
\mathbb{R}^2,
\qquad
a=1,\dots,N_i^{\mathrm{cell}}.
```

The journal version uses HoVer-Net for nuclei detection, pretrained on MoNuSeg,
then extracts node features using patches centered at nuclei centroids through a
ResNet backbone, plus normalized spatial coordinates.

The cell feature matrix is:

```math
H_i^{\mathrm{cell}}
=
\begin{bmatrix}
h_{i1}^{\mathrm{cell}}\\
\vdots\\
h_{iN_i^{\mathrm{cell}}}^{\mathrm{cell}}
\end{bmatrix}
\in
\mathbb{R}^{N_i^{\mathrm{cell}}\times d_{\mathrm{cell}}}.
```

The short HACT-Net paper used hand-crafted morphology, texture, and spatial
features. It states 18 features per nucleus. The later journal version shifts
the implementation toward CNN-based nucleus-centered features plus spatial
features. The mathematical graph object is the same:

```math
\text{cell node}
=
\text{detected nucleus with morphology and position}.
```

## Cell Edges

Cell edges are spatial proximity edges.

Define the Euclidean image-space distance:

```math
d_i^{\mathrm{cell}}(a,b)
=
\left\|
c_{ia}^{\mathrm{cell}}
-
c_{ib}^{\mathrm{cell}}
\right\|_2.
```

Let:

```math
\mathcal{K}_k^{\mathrm{cell}}(a)
```

be the set of the `k` nearest cell nodes to cell `a` under this distance.

The construction is thresholded kNN:

```math
(a,b)\in E_i^{\mathrm{cell}}
\quad
\Longleftrightarrow
\quad
b\in\mathcal{K}_k^{\mathrm{cell}}(a)
\;\text{and}\;
d_i^{\mathrm{cell}}(a,b)<d_{\min}.
```

The HACT-Net 2020 paper reports:

```math
k=5,
\qquad
d_{\min}=50\;\text{pixels}.
```

At the stated scanner resolution:

```math
50\;\text{pixels}
\times
0.25\;\mu\mathrm{m}/\text{pixel}
=
12.5\;\mu\mathrm{m}.
```

The paper defines undirected graphs with symmetric adjacency. A conservative
matrix reading is therefore the symmetrized support:

```math
A_{iab}^{\mathrm{cell}}
=
1
\quad
\Longleftrightarrow
\quad
(a,b)\in E_i^{\mathrm{cell}}
\;\text{or}\;
(b,a)\in E_i^{\mathrm{cell}}.
```

This is not a feature-similarity graph. Two cells communicate because they are
spatially nearby.

## Tissue Graph

The tissue graph is:

```math
G_i^{\mathrm{tissue}}
=
\left(
V_i^{\mathrm{tissue}},
E_i^{\mathrm{tissue}},
H_i^{\mathrm{tissue}}
\right).
```

Each tissue node is a tissue region, not a patch and not a cell.

The paper constructs tissue regions by:

```text
low-magnification oversegmentation
SLIC superpixels
merge neighboring similar superpixels
feature extraction for merged tissue regions
region adjacency graph
```

Let the resulting tissue regions be:

```math
R_{i1},\dots,R_{iN_i^{\mathrm{tissue}}}.
```

Their centroids are:

```math
c_{ib}^{\mathrm{tissue}}
\in
\mathbb{R}^2.
```

The tissue feature matrix is:

```math
H_i^{\mathrm{tissue}}
=
\begin{bmatrix}
h_{i1}^{\mathrm{tissue}}\\
\vdots\\
h_{iN_i^{\mathrm{tissue}}}^{\mathrm{tissue}}
\end{bmatrix}
\in
\mathbb{R}^{N_i^{\mathrm{tissue}}\times d_{\mathrm{tissue}}}.
```

The short HACT-Net paper describes color and texture attributes, random-forest
feature selection, and 26-dimensional tissue-region representations. The
journal version describes CNN features from superpixel-centered patches,
averaged into merged tissue regions, plus normalized spatial centroids.

Again, the invariant mathematical object is:

```math
\text{tissue node}
=
\text{merged tissue region with morphology, texture, and position}.
```

## Tissue Edges

Tissue edges are region adjacency edges.

Let:

```math
R_{ia}\sim R_{ib}
```

mean that two tissue regions are adjacent in the region adjacency graph.

Then:

```math
A_{iab}^{\mathrm{tissue}}
=
1
\quad
\Longleftrightarrow
\quad
R_{ia}\sim R_{ib}.
```

This is a different edge semantics from the cell graph.

Cell graph:

```text
nearby nuclei under thresholded kNN
```

Tissue graph:

```text
neighboring segmented tissue regions under RAG adjacency
```

The tissue graph can capture the spatial organization of segmented regions
that correspond to structures such as epithelium, stroma, necrosis, lumen, and
other tissue parts. The tissue nodes are still segmented and merged regions
with features; they are not guaranteed semantic labels.

## Construction As A Model Assumption

The full construction map is:

```math
\Gamma_{\mathrm{HACT}}:
I_i
\longmapsto
\left(
H_i^{\mathrm{cell}},
A_i^{\mathrm{cell}},
H_i^{\mathrm{tissue}},
A_i^{\mathrm{tissue}},
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\right).
```

This map is not learned end to end in the way a raw-pixel transformer would be.
It contains hand-specified operations:

```text
nuclei detector
centroid extraction
kNN plus distance threshold
superpixel segmentation
superpixel merging
region adjacency graph
feature extraction
cell-to-tissue assignment
```

The graph neural network learns after these choices have already defined the
available geometry.

## Inductive Bias

The construction uses this graph-support prior:

```math
\text{possible local interaction support}
\approx
\text{spatial proximity or tissue adjacency}.
```

For cells:

```math
e_{ab}^{\mathrm{cell}}
\text{ exists because }
c_{ia}^{\mathrm{cell}}
\text{ and }
c_{ib}^{\mathrm{cell}}
\text{ are nearby}.
```

For tissue regions:

```math
e_{ab}^{\mathrm{tissue}}
\text{ exists because }
R_{ia}
\text{ and }
R_{ib}
\text{ touch}.
```

This is a support prior, not a calibrated interaction-strength model. The model
can learn weights on top of this support, but it cannot pass messages along
edges that construction removed.

## What Construction Can Destroy

Two TRoIs can have similar downstream HACT graphs even if their pixels differ:

```math
\Gamma_{\mathrm{HACT}}(I_i)
=
\Gamma_{\mathrm{HACT}}(I_j),
\qquad
I_i\neq I_j.
```

Then every HACT-Net using only this representation must produce the same output:

```math
F_\theta
\left(
\Gamma_{\mathrm{HACT}}(I_i)
\right)
=
F_\theta
\left(
\Gamma_{\mathrm{HACT}}(I_j)
\right).
```

So the representation is interpretable because it is entity-based, but it is
also lossy because only extracted entities and edges survive.
