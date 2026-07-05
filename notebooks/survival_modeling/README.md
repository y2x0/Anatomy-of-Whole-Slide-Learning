# Survival Modeling

This notebook family asks one question:

```text
How is risk represented?
```

The answer is not always a single scalar. A survival model may output a risk
score, a discrete hazard vector, a continuous hazard function, a survival curve,
a density over event times, or cause-specific risks.

For whole-slide learning, the survival head sits on top of a slide
representation:

```math
S_i
\longrightarrow
H_i=\{h_{ij}\}_{j=1}^{n_i}
\xrightarrow{\mathcal C}
\widetilde H_i
\xrightarrow{\mathcal R}
z_i
\xrightarrow{\mathcal H}
\text{risk object}.
```

The first three families are:

```text
cox_family/
    risk is a scalar ordering function

discrete_time_hazards/
    risk is a vector of interval-wise event probabilities

continuous_time_hazards/
    risk is a time-indexed hazard, survival curve, or event-time density
```

The dense math pass adds the derivation kernels:

```text
cox_family/
    counting-process derivation
    score, Hessian, and Breslow baseline recovery
    exact WSI statistic optimized by Cox heads

discrete_time_hazards/
    hazard/PMF/survival algebra
    masked likelihood and gradients
    ranking, calibration, and horizon-risk functionals

continuous_time_hazards/
    monotone cumulative-hazard parameterizations
    mixture survival algebra
    time-conditioned WSI readouts and gradients
```

The notes are derivation-first. Each family should make explicit:

1. What object the model predicts.
2. Which likelihood or surrogate is optimized.
3. What statistic of the slide representation survives.
4. Which censoring assumptions are being used.
5. Which failure modes follow from the math.

## Core Observed Data

For patient or slide indexed by \(i\):

```math
T_i = \text{event time},
\qquad
C_i = \text{censoring time},
```

but we observe only

```math
X_i = \min(T_i,C_i),
\qquad
\delta_i = \mathbf 1[T_i \le C_i].
```

The pair \((X_i,\delta_i)\) is the survival label. Whole-slide pathology adds
the high-dimensional object \(S_i\), usually represented through patch features
\(H_i\).

## References To Anchor The First Pass

- Cox. "Regression Models and Life-Tables." JRSS-B 1972.
- Katzman et al. "DeepSurv: Personalized Treatment Recommender System Using a
  Cox Proportional Hazards Deep Neural Network." 2016.
  https://arxiv.org/abs/1606.00931
- Ching et al. "Cox-nnet: An artificial neural network method for prognosis
  prediction of high-throughput omics data." PLOS Computational Biology 2018.
- Gensheimer and Narasimhan. "A Scalable Discrete-Time Survival Model for
  Neural Networks." 2018. https://arxiv.org/abs/1805.00917
- Lee et al. "DeepHit: A Deep Learning Approach to Survival Analysis With
  Competing Risks." AAAI 2018.
  https://ojs.aaai.org/index.php/AAAI/article/view/11842
- Nagpal et al. "Deep Cox Mixtures for Survival Regression." MLHC 2021.
- Chen et al. "Whole Slide Images are 2D Point Clouds: Context-Aware Survival
  Prediction using Patch-based Graph Convolutional Networks." MICCAI 2021.
  https://arxiv.org/abs/2107.13048
- Chen et al. "Multimodal Co-Attention Transformer for Survival Prediction in
  Gigapixel Whole Slide Images." ICCV 2021.
