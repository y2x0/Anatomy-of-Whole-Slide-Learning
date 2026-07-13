# Transfer Geometry and Foundation Model Evaluation

## 1. Linear Probe

For frozen embeddings `h`, a linear probe tests

```math
\widehat y=\mathrm{softmax}(Wh+b).
```

Good probe performance means the task is linearly accessible under the chosen
sample and label distribution. It does not prove the representation is
causally or clinically complete.

## 2. Geometry-Dependent Evaluation

Different downstream functionals probe different geometry:

```math
\begin{aligned}
\text{classification:}&\quad \text{linear separability},\\
\text{retrieval:}&\quad \text{neighbor label coherence},\\
\text{survival:}&\quad \text{risk ordering and calibration},\\
\text{segmentation:}&\quad \text{local spatial information},\\
\text{multimodal:}&\quad \text{cross-modal alignment}.
\end{aligned}
```

No single benchmark certifies all five.

## 3. Dataset Leakage

If related patches from one patient appear in train and test,

```math
I(H_{\mathrm{train}};H_{\mathrm{test}}\mid\text{patient})
```

can be large. Patient-level splits are required for meaningful pathology
transfer claims.

## 4. Stability and Shift

Report performance as a function of stain, scanner, institution, cancer type,
and rare morphology. A foundation representation should be evaluated both on
average transfer and on the failure modes its pretraining contract predicts.

