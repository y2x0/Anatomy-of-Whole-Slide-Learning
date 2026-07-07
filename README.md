# The Anatomy of Whole Slide Learning: A Mathematical Taxonomy of Aggregation and Representation

To make the mathematical shape of whole-slide learning legible, so researchers
can understand, compare, and design methods rather than just implement them.

This repository is a mathematical reference for whole-slide learning in
computational pathology. Its purpose is to make the field legible: why methods
work, what assumptions they make, which mathematical object they preserve, and
which failure mode follows from each inductive bias.

Most computational pathology resources sit at one of two extremes. Some are code
tutorials: here is how to run CLAM, train a MIL model, or extract patch features.
Others are dense papers whose architectures sound different even when their
mathematical structure is nearly the same. This project sits between those
extremes. It is for researchers and early PhD students who want a clean
mathematical map of the field.

The organizing principle is that most whole-slide methods can be placed inside a
common decomposition:

```math
\widetilde H=\mathcal C(H;G,S),
\qquad
z=\mathcal R(\widetilde H),
\qquad
\widehat y=\mathcal H(z).
```

Here:

- `H` is the set, sequence, graph, hierarchy, distribution, or memory-derived
  collection of patch representations.
- `G` is optional geometry.
- `S` is optional sequence order.
- `\mathcal C` is the context operator.
- `\mathcal R` is the readout operator.
- `z` is the surviving slide statistic.
- `\mathcal H` is the task head, such as classification, survival, retrieval,
  or multimodal prediction.

The goal is that a reader can take any paper in whole-slide learning and answer:

1. What mathematical object represents the slide?
2. How is context injected between patches?
3. How are many patch representations reduced to one slide statistic?
4. What statistic survives the reduction?
5. What inductive bias does the method rely on?
6. What failure mode does that bias create?
7. What unexplored design axis might lead to a new method?

## Scope

The repository will grow as a set of deep mathematical notes. Each note should
do real explanatory work: define the objects, derive the forward map, identify
the surviving statistic, explain the inductive bias, and make the failure mode
mathematically visible.

The existing MIL material will eventually be encompassed here, but not as a
public routing map and not as shallow taxonomy stubs. The goal is to build the
math notes first.

## Notebook Families

- `notebooks/survival_modeling/`: how risk is represented, optimized, and
  evaluated.
- `notebooks/slide_representations/`: what mathematical object represents a
  whole slide.
- `notebooks/pooling_operators/`: what information survives aggregation.
