# Hierarchical MIL Failure Modes And Counterexamples

## 1. Bad partitions create a wrong inductive bias

A hierarchy assumes that children grouped under one parent should be summarized
together. If a true signal crosses the partition boundary, local coarsening can
destroy it before any coarse context is applied.

Consider four fine instances with two possible partitions:

`math
P_A:
\quad
\{x_1,x_2\},\{x_3,x_4\},
\qquad
P_B:
\quad
\{x_1,x_3\},\{x_2,x_4\}.
`

Suppose the label depends on the pair interaction between x_1 and x_3. A
within-parent operator followed by a parent-level mean can detect the relation
under P_B but may discard the required pairing under P_A. The same flat set with
a different parent map is a different hierarchical input.

This is not an optimization failure. It is a representation mismatch.

## 2. Coarsening collision

Let a parent use sum pooling. The two child configurations

`math
\{u+\delta,u-\delta\}
\quad\text{and}\quad
\{u,u\}
`

have the same parent state 2u. Any later model that sees only the parent token
must assign them the same output. The lost contrast is invisible regardless of
the capacity of the coarse transformer.

Mean pooling has the same collision after normalization. Attention changes the
collision set but can create a new one whenever it selects the same effective
convex combination.

## 3. Selection and rare positives

Suppose a positive slide contains a small positive fraction p and DTFD creates
pseudo-bags of size m. The chance that a pseudo-bag contains no positive patch is

`math
(1-p)^m.
`

Inherited positive labels then train Tier 1 on examples that may be visually
negative. Increasing the number of pseudo-bags creates more virtual bags but
reduces m, increasing this noise probability. The choice R is therefore a
statistical tradeoff, not a free sample-size multiplier.

Top-k readouts create the opposite risk: if a true positive is not selected,
later layers never see it. A useful evaluation should report recall of positive
regions under the selection rule, not only slide AUC.

## 4. Geometry leakage and split leakage

A hierarchy can leak information if parent construction uses annotations,
patient-level labels, or test-slide statistics. Examples include:

`text
- using tumor masks to define regions before a supposedly weakly supervised
  experiment;
- fitting a learned partition jointly on all slides before the train/test split;
- using slide-level outcome information to choose parent regions;
- allowing overlapping crops from one physical tissue area across data splits.
`

Coordinate-based partitions are not automatically leakage, but label-dependent
partitions are supervision and must be declared.

## 5. Hierarchy-induced order dependence

If a parent map is produced by a sequence-sensitive algorithm, the hierarchy may
change under a permutation of the same fine bag. For a set-derived hierarchy,
the construction should satisfy

`math
\mathcal P(QH,QX)
=
Q_{\mathrm{parent}}\mathcal P(H,X)
`

for every admissible permutation Q. Otherwise the model has an accidental
ordering inductive bias in addition to its intended spatial bias.

## 6. Complexity is conditional

For n fine units, R regions, and m=n/R units per region, hierarchical attention
has rough cost

`math
R m^2+R^2.
`

The comparison against flat n^2 is favorable only if R is much smaller than n
and the partition is balanced. One giant region gives a flat-scale quadratic
term; R close to n gives a near-flat region-level term. Padding every region to
a common maximum can also turn a balanced theoretical cost into a worst-case
implementation cost.

## 7. Explanations stop at boundaries

A high-scoring parent is not a proof that every child mattered. The final
effective weight under two attention levels is

`math
\omega_a
=
\beta_{\pi(a)}\alpha_{a,\pi(a)}.
`

If a transformer or graph operates inside a parent, there may be no unique
decomposition into child contributions at all. Explanations should therefore
state whether they show:

`text
- parent selection;
- local attention;
- gradient attribution to the final output;
- perturbation effect after recomputing all upstream summaries.
`

These are different explanatory objects.

## 8. Fixed versus learned hierarchy

A fixed grid, random pseudo-bag partition, or cell-to-tissue assignment has a
different uncertainty profile from a learned router. A learned hierarchy can
adapt to morphology but may collapse to trivial assignments:

`math
A_{ab}\approx 1
\quad\text{for one parent }b
`

or diffuse routing:

`math
A_{ab}\approx \frac{1}{R}
\quad\text{for all }b.
`

The first reduces effective parent diversity; the second erases region identity.
Assignment entropy, parent occupancy, and stability under perturbations should be
measured when routing is learned.

## 9. Failure-mode checklist

`text
1. Verify parent-map equivariance under unit permutation.
2. Measure the fraction of signal-bearing children removed by each boundary.
3. Report parent occupancy and the effective number of selected parents.
4. Compare balanced and adversarial partitions.
5. Audit partition construction for label and split leakage.
6. Compare final-output attribution with local attention maps.
`
