# HACT And The Hierarchical Graph MIL Boundary

Sources:

```text
Pati et al., HACT-Net: A Hierarchical Cell-to-Tissue Graph Neural Network for
Histopathological Image Classification.
https://arxiv.org/abs/2007.00584

Pati et al., Hierarchical graph representations in digital pathology.
https://arxiv.org/abs/2102.11057
```

## 1. Three meanings of hierarchy

The word hierarchy hides three distinct operators:

```math
\text{DTFD:}
\quad
\text{random instance partition}
\longrightarrow
\text{pseudo-bag features}
\longrightarrow
\text{slide MIL};
```

```math
\text{HIPT:}
\quad
\text{nested image windows}
\longrightarrow
\text{nested transformer tokens}
\longrightarrow
\text{slide representation};
```

```math
\text{HACT:}
\quad
\text{cell graph}
\longrightarrow
\text{cell-to-tissue assignment}
\longrightarrow
\text{tissue graph}.
```

Only the third makes the fine-to-coarse relation an explicit graph object.
DTFD's parent relation is a training partition. HIPT's parent relation is a
spatial tokenization contract.

## 2. HACT as an explicit hierarchical graph

For cells and tissue regions, HACT supplies

```math
\mathcal G_i
=
\left(
H_i^{\mathrm{cell}},
A_i^{\mathrm{cell}},
H_i^{\mathrm{tissue}},
A_i^{\mathrm{tissue}},
B_i
\right),
```

where B_i is the binary cell-to-tissue assignment matrix. A cell graph context
operator produces

```math
\widetilde H_i^{\mathrm{cell}}
=
\mathcal C_{\theta}^{\mathrm{cell}}
\left(
H_i^{\mathrm{cell}},A_i^{\mathrm{cell}}
\right).
```

The assignment readout is

```math
U_i^{\mathrm{cell}\to\mathrm{tissue}}
=
B_i^{\mathsf T}\widetilde H_i^{\mathrm{cell}}.
```

The tissue state is initialized by concatenating original tissue features and
the transferred cell statistic:

```math
H_i^{\mathrm{tissue},0}
=
\left[
H_i^{\mathrm{tissue}}
\middle\Vert
U_i^{\mathrm{cell}\to\mathrm{tissue}}
\right].
```

A tissue graph operator then supplies cross-region context before the final
tissue-node readout.

The HACT graph note derives these operators in full. The point here is their
MIL interpretation: HACT replaces one flat patch bag with a structured
fine-to-coarse bag whose assignment map is visible to the model.

## 3. Same C/R/G/S axes, different geometry

```text
DTFD:
    C = attention in two tiers
    R = pseudo-bag selection or attention, then parent-slide attention
    G = random pseudo-bag membership
    S = inherited slide labels at both tiers

HIPT:
    C = transformer self-attention within nested windows
    R = [CLS] token extraction at each scale
    G = non-overlapping spatial nesting and positional structure
    S = DINO pretraining plus downstream slide labels

HACT:
    C = graph message passing at cell and tissue scales
    R = assignment sum followed by tissue-node readout
    G = explicit cell graph, tissue graph, and assignment matrix
    S = slide labels, with pseudo-types available upstream
```

The same four-way decomposition is useful because the context operator and
geometry are independent design choices. A transformer can run on a hierarchy;
a graph can run on a flat patch set; a pseudo-bag can use attention without
being spatial.

## 4. What can and cannot be merged

DTFD and HACT both have a child-to-parent aggregation, but they differ in what
the parent means. In DTFD, a pseudo-bag is sampled to increase the number of
training examples and its inherited label may be noisy. In HACT, a tissue node
is an explicit coarse entity and the assignment map is part of the graph.

HIPT and HACT both preserve multiscale organization, but HIPT's within-window
context is token self-attention while HACT's is graph message passing. HIPT
may drop fine tokens at [CLS] boundaries; HACT's assignment sum drops
within-tissue contrasts unless later fine states are retained.

A unified diagram is therefore not a claim of equivalence:

```math
\begin{array}{c}
\text{fine units}
\\
\downarrow\ \mathcal C_0
\\
\text{child states}
\\
\downarrow\ \mathcal R_0
\\
\text{parent states}
\\
\downarrow\ \mathcal C_1
\\
\text{coarse states}
\\
\downarrow\ \mathcal R_1
\\
\text{slide statistic}
\end{array}
```

Each paper instantiates different C, R, and G.

## 5. Modeling rule

A new hierarchical MIL proposal should state four objects separately:

```text
1. the parent map or routing rule;
2. the context operator before and after coarsening;
3. the statistic retained at each boundary;
4. the supervision attached to each level.

Without those declarations, "hierarchical" only says that the implementation
has more than one resolution.
```
