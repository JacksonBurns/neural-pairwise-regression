---
title: Neural Pairwise Regression
author: Jackson Burns
header-includes: |
    \usetheme{Custom}
    \addtobeamertemplate{footnote}{}{\vspace{3ex}}
    \usepackage{tikz}
    \usetikzlibrary{positioning}
    \usepackage{amsmath}
bibliography: refs.bib
citation-style: apa
note: |
 These slides can be compiled to other formats (i.e. pdf) using pandoc.
 See https://pandoc.org/MANUAL.html#structuring-the-slide-show for usage details.

 This file was used as a starting point:
 https://pandoc.org/chunkedhtml-demo/10-slide-shows.html

 I usually use this command to compile the slides into a PDF for showing as a slideshow:
   pandoc --citeproc -t beamer slides.md -o slides.pdf
 (ran from the meta/presentation directory in neural-pairwise-regression).
---

## Agenda

\tableofcontents

# Regressing Hard Targets

## Canonical Regression Problem

 - Many fields seek __surrogate models__ for arbitrary quantities of interest
    - solubility as a function of a molecular embedding instead of experiment
    - band gap as a function of formula instead of expensive simulation
 - When target $y$ is continuous this is a regression problem $y=f(x; \theta)$
    - $x$ is some static or learned _embedding_ i.e. a vector of scalars
    - $f(...; \theta)$ is an arbitrary model with parameters

## Regression Methods

 - Terminology is loose[^1] but broadly we use:
   - Statistical Modeling (SM): (regularized) linear methods
   - Machine Learning (ML): Decision Trees, Neighbors-Based, and _Ensembles_
   - Artificial Intelligence (AI): Neural Networks (NNs)
 - Broadly speaking, moving down that list:
   - requires more data to fit an accurate model because $f$ has more parameters $\theta$
   - offers a higher accuracy 'ceiling' and can fit 'harder' datasets
   - reduces interpretability

[^1]: Some use these interchangeably i.e. linear regression = AI

## Canonical Neural Network

::: columns

:::: column

 - NNs are especially popular
    - easy to train with high-level languages
    - trivial hardware acceleration
    - interpretability drawbacks secondary to improved performance
 - Generality reduces the 'time-to-model'

::::

:::: column

\begin{center}
\begin{tikzpicture}[
    node distance=1cm and 1cm,
    every node/.style={draw, circle, minimum size=0.7cm, inner sep=0pt}
  ]

  % Input layer
  \foreach \i in {1,2} {
    \node (E\i) at (0,-\i) {E$_{\i}$};
  }
  \node (E3) at (0, -3) {E$_{n}$};

  \foreach \i in {1,2} {
    \node (H1\i) at (2,-\i) {H$_{d,\i}$};
  }
  \node (H13) at (2,-3) {H$_{d,h}$};

  % Output layer
  \node (O) at (4,-2) {O};

  % Connections
  \foreach \i in {1,2,3} {
    \foreach \j in {1,2,3} {
      \draw[->] (E\i) -- (H1\j);
    }
  }
  \foreach \i in {1,2,3} {
    \draw[->] (H1\i) -- (O);
  }

\end{tikzpicture}
\end{center}

![Schematic of typical Feedforward Neural Network with input encoding $E$ of dimension $n$, Hidden Layers $H$ of height $h$ and depth $d$, and Output $O$. Arbitrary activation functions (i.e. ReLU) at output of each node not shown.\label{fnn}](./figures/blank.png)

::::

:::


## Data Limitations

 - Large, high-quality datasets are rare
    - _especially_ in science domains where experiments are expensive
    - negative ramifications for coverage of feature space
 - Mapping structure $\rightarrow$ target is _hard_
    - static and learned representations are _very_ high-dimensional, $O(100)+$ features
    - relationships are often unpredictably non-linear (i.e. Activity Cliffs, some examples in [@activity_cliffs])

\begin{alertblock}{The Wicked Problem}
Neural networks are well-suited for our problems but we lack sufficient data to fit them accurately.
\end{alertblock}

# Reformulating Regression

## Pairwise Regression

 - We can address The Wicked Problem by recasting our regression problem to difference prediction:
    - instead of predicting the output directly: $y=f(x;\theta)$
    - we predict the _difference_ in target for two inputs at a time: $y_1-y_2=\Delta_{1,2}=f(x_1,x_2;\theta)$
 - This addresses our key constraint
    - quadratic increase in the amount of training data
 - Offers additional benefits
    - ensemble of predictions during inference provides well-calibrated uncertainty estimates [@tnnr]
    - theoretical evidence that inference is easier (OOS vs OOD) [@bilinear_transduction]

## Pairwise Architecture

::: columns

:::: column

 - Choose the most _basic_ approach to achieve this mapping
    - inputs are directly concatenated
    - keeps SHAP-ability [@shap]
    - leans on NN Universal Approximation Theorem [^2]

::::

:::: column

\begin{center}
\begin{tikzpicture}[
    node distance=1cm and 1cm,
    every node/.style={draw, circle, minimum size=0.7cm, inner sep=0pt}
  ]

  % Input layer
  \node (E11) at (0,-1+0.25) {E$_{1,1}$};
  \node (E12) at (0, -2+0.25) {E$_{1,n}$};
  \node (E21) at (0,-3-0.25) {E$_{2,1}$};
  \node (E22) at (0, -4-0.25) {E$_{2,n}$};

  \foreach \i in {1,2} {
    \node (H1\i) at (2,-0.5-\i) {H$_{d,\i}$};
  }
  \node (H13) at (2,-0.5-3) {H$_{d,h}$};

  % Output layer
  \node (O) at (4,-0.5-2) {O};

  % Connections
  \foreach \i in {1,2} {
    \foreach \j in {1,2,3} {
      \draw[->] (E1\i) -- (H1\j);
      \draw[->] (E2\i) -- (H1\j);
    }
  }
  \foreach \i in {1,2,3} {
    \draw[->] (H1\i) -- (O);
  }

\end{tikzpicture}
\end{center}

![Schematic of a Neural Pairwise Regressor with concatenated input encodings $E_{i,n}$\label{npr}](./figures/blank.png)

::::

:::

[^2]: Closed-form width and depth requirements for ReLU networks known [@shen_uat]

## Need for Naivete

Empirical exploration suggests avoiding __inductive bias__ in NN architecture

  - Literature has explored alternative design in the general case (see next slide)
  - My previous work with collaborators explored it specifically for solubility [@Attia_Burns_Doyle_Green_2024]

## Existing Alternatives

There are many ways to inject bias into the model:

 - [@tynes_padre]: input $x_1 \oplus x_2 \oplus (x_1 - x_2)$ and use ML
    - inductive bias in input distance (i.e. unreduced manhattan) is probably unhelpful
 - [@bilinear_transduction]: include an operation in later layers (i.e. subtraction) to enforce self 'loop'
    - do we really want to enforce this property?
 - [@tnnr]: additional latent layers for each input
    - makes it _possible_ for network to learn a different embedding for each input 'branch' (bad!)
    - [@tnnr_2] moved away from this and has no embedding layers
    - a _shared_ learnable embedding would be OK (foreshadowing)

## The Bitter Lesson

Rich Sutton famously pointed out in a blog post [^3] that in the Computer Vision world:

 - architectures based on human intuition for how to process images failed
 - Convolutional NNs (which have almost _no_ inductive bias) dominate the space 

\begin{alertblock}{The Bitter Lesson}
The most general, scalable approach is the most effective.
\end{alertblock}

One criticism of this argument is that hardware is no longer following Moore's law ... but we (and 99.9% of people) aren't looking at a scale where that matters.

[^3]: Representative publication: [@bitter_lesson_analysis]

## 'Physics' Constraints

There are a number of cases of the 'cycle closure' rule that we _might_ want to enforce:

 - Arbitrary loop: $\Delta_{1,2} + \Delta_{2,n} + ... + \Delta_{n,1} = 0$
    - Get this 'for free' with an accurate model
 - Pairwise 'loop': $\Delta_{1,2} + \Delta_{2,1} = 0$
    - Requires positional invariance of inputs in $f$
 - Self 'loop': $\Delta_{1,1} = 0$
    - Inductive bias in model architecture (i.e. subtract the latent representations) can explicitly enforce this

## Visualizing Training Augmentation
We can take our inputs representations and $x$:
$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}, \quad
\mathbf{x}^T = \begin{bmatrix} x_1 & x_2 & \cdots & x_n \end{bmatrix} 
$$
And generate all of the pairs with this matrix:
$$
\begin{bmatrix} 
    x_1 \oplus x_1 & x_1 \oplus x_2 & \cdots & x_1 \oplus x_n \\
    x_2 \oplus x_1 & x_2 \oplus x_2 & \cdots & x_2 \oplus x_n \\
    \vdots & \vdots & \ddots & \vdots \\
    x_n \oplus x_1 & x_n \oplus x_2 & \cdots & x_n \oplus x_n 
\end{bmatrix}
$$

## Augmentation Impacts

::: columns

:::: column

Each part of the matrix enforces a different property by being present during training:

 - Diagonal: Self loop [^5]
 - SUT _or_ SLT [^4]: general loop consistency
 - SUT _and_ SLT: Pairwise 'loop'

::::

:::: column

$$
\begin{bmatrix} 
    x_1 \oplus x_1 & x_1 \oplus x_2 & \cdots & x_1 \oplus x_n \\
    x_2 \oplus x_1 & x_2 \oplus x_2 & \cdots & x_2 \oplus x_n \\
    \vdots & \vdots & \ddots & \vdots \\
    x_n \oplus x_1 & x_n \oplus x_2 & \cdots & x_n \oplus x_n 
\end{bmatrix}
$$

::::

:::

[^4]: Strictly Upper/Lower Triangular
[^5]: _Very_ underrepresented (~1:n)

## Inference Augmentation

Given a vector of known $y$ values corresponding to the training data, i.e. anchors $y^a$

$$
\mathbf{y^a} = \begin{bmatrix} y^a_1 & y^a_2 & \cdots & y^a_n \end{bmatrix}^T, \quad
\mathbf{y} = \begin{bmatrix} y_1 & y_2 & \cdots & y_m \end{bmatrix}^T
$$

We can run inference against all of the anchors using this 'vector product':

$$
\mathbf{y^a} \mathbf{y}^T = \begin{bmatrix} y^a_1 \\ y^a_2 \\ \vdots \\ y^a_n \end{bmatrix} 
\begin{bmatrix} y_1 & y_2 & \cdots & y_m \end{bmatrix} =
\begin{bmatrix} 
    y^a_1 - y_1 & y^a_1 - y_2 & \cdots & y^a_1 - y_m \\
    y^a_2 - y_1 & y^a_2 - y_2 & \cdots & y^a_2 - y_m \\
    \vdots & \vdots & \ddots & \vdots \\
    y^a_n - y_1 & y^a_n - y_2 & \cdots & y^a_n - y_m 
\end{bmatrix}
$$

##

But you can also augmented in the _other_ direction _if_ you have enforced pairwise loop consistency during training, i.e.

$$
\mathbf{y^a}^T \mathbf{y} = \begin{bmatrix} y^a_1 & y^a_2 & \vdots & y^a_n \end{bmatrix} 
\begin{bmatrix} y_1 \\ y_2 \\ \cdots \\ y_m \end{bmatrix} =
\begin{bmatrix} 
    y_1 - y^a_1 & y_1 - y^a_2 & \cdots & y_1 - y^a_m \\
    y_2 - y^a_1 & y_2 - y^a_2 & \cdots & y_2 - y^a_m \\
    \vdots & \vdots & \ddots & \vdots \\
    y_n - y^a_1 & y_n - y^a_2 & \cdots & y_n - y^a_m 
\end{bmatrix}
$$

Mapping back to absolute predictions can be done with averaging.

## Augmentation Alternatives

Running inference with this approach is an open research area:

 - ML theorists suggest:
    - 'full' augmentation and averaging [@tnnr]
    - various post-training weighting schemes to identify 'best' anchors [@pairwise_weighted]
    - use K-nearest anchors in embedding space [@bilinear_transduction]
 - Chemistry practitioners seem to ignore it:
    - train and infer with one direction [@tynes_padre]
    - check self loop consistency after-the-fact [@deepdelta]

# Initial Investigation

## Implementing `nepare`

The `nepare` package obfuscates the implementation details of the decisions previously discussed:

```python
npr = NeuralPairwiseRegressor(
  input_size=2,
  hidden_size=50,
  num_layers=3,
)
```

`NeuralPairwiseRegressor` is a subclass of a `FeedforwardNeuralNetwork` which is useful for comparison.

## Ease of Augmentation

Augmenting training can easily be configured:

```python
training_dataset = PairwiseAugmentedDataset(
  X[train_idxs],
  y[train_idxs],
  how='full',  # one of: 'full', 'ut', 'sut'
)
```

## Automatic Anchoring

Validation and inference anchoring are also handled behind the scenes:

```python
validation_dataset = PairwiseAnchoredDataset(
  X[train_idxs],
  y[train_idxs],
  X[val_idxs],
  y[val_idxs],
  how='full',  # one of: 'full', 'half'
)
predict_dataset = PairwiseInferenceDataset(
  X[train_idxs],
  y[train_idxs],
  X_unknown,
  how='full',  # one of: 'full', 'half'
)
```

## Fitting Arbitrary Surfaces

_demo notebook_

## Aqueous Solubility Prediction

Here we use `mordred(-community)` as the encoder.

<!-- | Customer / Source | Arnhem [euro/ton] |  Gouda [euro/ton] | Demand [tons] |
| :---: | :--: | :--: | :--: |
| London | n/a | 2.5 | 125 |
| The Hague | 1.4 | 0.8 | 200 |
| **Supply [tons]** | 550 tons | 700 tons | n/a | -->

# Subsequent Steps

## Learned Embeddings

Up to this point I have been quietly assuming that some function like this exists:

\begin{center}
\begin{tikzpicture}[
    node distance=1.5cm and 1.5cm,
    every node/.style={draw, minimum width=2cm, minimum height=1cm, align=center}
  ]

  % Input box
  \node (Input) at (0,0) {Input};
  
  % Encoder node
  \node[draw, circle, minimum size=0.7cm] (Encoder) at (3,0) {Encoder};
  
  % Encoder output nodes
  \node[draw, circle, minimum size=0.7cm] (E1) at (6,1) {E1};
  \node[draw, circle, minimum size=0.7cm] (E2) at (6,0) {E2};
  \node[draw, circle, minimum size=0.7cm] (E3) at (6,-1) {E$_{n}$};
  
  % Arrows
  \draw[->] (Input) -- (Encoder);
  \draw[->] (Encoder) -- (E1);
  \draw[->] (Encoder) -- (E2);
  \draw[->] (Encoder) -- (E3);

\end{tikzpicture}
\end{center}

We may want to _learn_ this embedding rather than keep it static

 - Fully differentiable architecture, can readily plug in modules like ChemProp's message passing
 - DeepDelta has done an incorrect version of this in which they had two separate MP blocks [@deepdelta]

## LearnedEmbeddingNPR

```python
class LearnedEmbeddingNPR(NeuralPairwiseRegressor)
    def __init__(self, ..., embedding_module)
```

\begin{center}
\begin{tikzpicture}[
    node distance=1cm and 1cm,
    every node/.style={draw, circle, minimum size=0.7cm, inner sep=0pt}
  ]

  % Input boxes
  \node[draw, rectangle, minimum width=2cm, minimum height=1cm, align=center] (Input1) at (0,1) {Input 1};
  \node[draw, rectangle, minimum width=2cm, minimum height=1cm, align=center] (Input2) at (0,-1) {Input 2};
  
  % Encoder box
  \node[draw, circle, minimum size=0.7cm] (Encoder) at (2,0) {Encoder};
  
  % Encoder output nodes
  \node (E11) at (4,1.5) {E$_{1,1}$};
  \node (E12) at (4,0.5) {E$_{1,n}$};
  \node (E21) at (4,-0.5) {E$_{2,1}$};
  \node (E22) at (4,-1.5) {E$_{2,n}$};
  
  % Arrows
  \draw[->] (Input1) -- (Encoder);
  \draw[->] (Input2) -- (Encoder);
  \draw[->] (Encoder) -- (E11);
  \draw[->] (Encoder) -- (E12);
  \draw[->] (Encoder) -- (E21);
  \draw[->] (Encoder) -- (E22);

  \foreach \i in {1,2} {
    \node (H1\i) at (6,2-\i) {H$_{d,\i}$};
  }
  \node (H13) at (6,-1) {H$_{d,h}$};

  % Output layer
  \node (O) at (8,0) {O};

  % Connections
  \foreach \i in {1,2} {
    \foreach \j in {1,2,3} {
      \draw[->] (E1\i) -- (H1\j);
      \draw[->] (E2\i) -- (H1\j);
    }
  }
  \foreach \i in {1,2,3} {
    \draw[->] (H1\i) -- (O);
  }

\end{tikzpicture}
\end{center}

## Uncertainty Calibration

I want to provide an in-depth analysis of how well calibrated the model is - need help getting started.

We have some previous work in this group that I am aware of.

I would especially like to know of any existing __frameworks__ for doing this so I can avoid re-implementing a bunch of metrics/methods/etc. from scratch.

# Supplementary Material

## Miscellaneous Thoughts

There are some applications where predicting property _differences_ is actually the typical application:
 - `confrank`: github.com/grimme-lab/confrank
 - all of the Relative Binding Free Energy (RBFE) research space
    - easier to predict difference in BFE than absolute
    - many tools exist to plan 'routes' of low-error difference calculations

Some ML theorists suggest one can enforce pairwise learning without actually changing the target:
 - `AdaPRL` train a normal NN but penalize pairwise distances in loss function (https://arxiv.org/abs/2501.05809)

## Cited Works

<!-- need below to make all the citations fit on the slide -->

\tiny
