# WSI Set Counterfactuals and HIPPO

HIPPO intervenes on the WSI bag while holding the trained model fixed.

## 1. Bag State Space

For slide bag

```math
B_i=\{h_{ij}\}_{j=1}^{n_i},
```

a deletion counterfactual selects subset `A`:

```math
B_i^{(-A)}=B_i\setminus A.
```

An addition counterfactual can augment the bag with patches from a specified
source:

```math
B_i^{(+A)}=B_i\uplus A.
```

The multiset union matters when repeated instances affect aggregation.

## 2. Minimum Decision-Flipping Subset

For thresholded decision `D`, the combinatorial ideal is

```math
A_i^{\star}
\in\arg\min_{A\subseteq B_i}|A|
\quad\text{subject to}\quad
D(B_i\setminus A)\ne D(B_i).
```

Exact search is exponential. Greedy removal ranks candidate patches by observed
score change and recomputes the model after each edit. Without submodularity,
greedy search has no general guarantee of minimum cardinality.

## 3. Hypothesis-Driven Interventions

If annotation or a detector defines tissue subset `T`, then

```math
\Delta_T
=F(B_i)-F(B_i\setminus T)
```

tests model necessity of the observed subset under deletion. Adding `T` to a
different bag tests a specified sufficiency-like transfer. Neither operation
creates the morphology that would replace removed tissue in a physically
realized slide.

## 4. Context Recalculation

For contextual MIL,

```math
F(B_i\setminus A)
```

must rerun attention, graph construction, transformer interactions, or
state-space recurrence. Freezing original weights estimates a different
quantity from deleting instances and recomputing the model.

## 5. Counterfactual Status

HIPPO provides exact model counterfactuals in bag space. It does not claim that
the edited bag is a biologically plausible alternative specimen. Its strength
is direct testing of set-level dependence without requiring a generative image
model.

