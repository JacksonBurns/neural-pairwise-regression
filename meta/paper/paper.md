---
title: "Pairwise Augmentation for Molecular Property Prediction"
subtitle: "Mini-Review and Initial Exploration"
author: Jackson Burns
date: 22 February 2025
geometry: margin=1in
bibliography: paper.bib
citation-style: journal-of-cheminformatics
colorlinks: true
note: |
 This paper can be compiled to other formats (pdf, word, etc.) using pandoc:
   pandoc --citeproc -s paper.md -o paper.pdf
 from the paper directory.
---

This short paper aims to do a few things:

 - formalize my (limited) literature review
 - describe my initial efforts into this approach (especially related to `polaris`)
 - briefly mention some of ideas for future directions

Much of this paper has been adapted from the corresponding presentation in `meta/presentation` - consider scrolling through those slides first to get an idea of what Neural Pairwise Regression is.

To summarize, I'll say that Pairwise Augmentation is an emerging regression technique with significant potential for small-dataset machine learning (i.e. nearly _all_ of chemistry).
At a high level, it recasts the canonical regression problem $y=f(x; \theta)$ into the difference-prediction problem $y_1-y_2=\Delta_{1,2}=f(x_1,x_2;\theta)$, offering more training data and a (possibly) easier task for improved extrapolation.
ML theorists are developing guidelines for using it while practitioners in the chemistry space are gathering empirical evidence of its efficacy.
My contribution is to explore the use of neural networks in this context and provide a high-quality Python library in doing so.
I've also run some initial benchmarks using the `polaris` library, and the results are promising.

# Background

Surrogate models are arbitrary mathematical models that replace 'expensive' alternatives.
In chemistry-world this could mean a solubility prediction model that circumvents the need to synthesize a drug candidate and actually run an experiment to determine its solubility.
This 'canonical' problem is therefore a search for some function $f$ with parameters $\theta$ which maps a vector representation of your input $x$ into your target property $y$, i.e. $y=f(x; \theta)$.

For the scope of this study I'm going to focus on regression problems where $y$ is continuous and (hopefully) approximately normally distributed.
There are an uncountable number of techniques, frameworks, and methods that could constitute $f$, but I'll just mention a few here[^1]:

 - Statistical Modeling (SM) such as (regularized) linear regression
 - Machine Learning (ML) such Decision Trees, Neighbors-Based, and _Ensembles_
 - Artificial Intelligence (AI) such Neural Networks (NNs)

[^1]: This terminology is _very_ loose, some use these interchangeably i.e. linear regression = AI

Broadly speaking, as one moves down that list one requires more data to fit an accurate model ($\theta$ increases) and results are less interpretable but one can fit 'harder' functions at better accuracy.
Science domains are therefore in a tough spot - our complex, highly non-linear problems would be perfect for these methods, but our datasets are often small and cover only a small subset of feature space.
We can counteract these challenges with pairwise augmentation.

Implementing pairwise augmentation starts by reformulating our canonical regression problem.
Using all of the same techniques from above, we will now take in two $x$'s simultaneously and predict the _difference_ in their target property value, i.e. $y_1-y_2=\Delta_{1,2}=f(x_1,x_2;\theta)$.
This reformulation allows us to then generate all pairwise combinations of our training data, increasing its amount by a power of 2.
During inference, we predict the property difference of an unknown $x$ to the known $x$'s in training to undo the reformulation, with the additional benefit of providing a well-calibrated prediction of the uncertainty [@tnnr].

This technique has simultaneously emerged in a few different places across science in the past few years with different thoughts on how it should be implemented:

 - [@tynes_padre]: Passed two inputs _and_ the difference between the inputs $x_1 \oplus x_2 \oplus (x_1 - x_2)$, used ML
 - [@bilinear_transduction]: Provided theoretical evidence that difference learning could make extrapolation easier by converting it from an out of support problem to an out of combination problem - also suggested including an operation within the model to enforce physics constraints.
 - [@tnnr]: First study to formalize pairwise regression as a subfield, demonstrated use of NNs as $f$.
    - Same authors later simplified their proposed architecture [@tnnr_2]

Their are many open questions[^2] about how to augment training, which physics constraints should be enforced, and how inference should be run(see [@pairwise_weighted]).
Perhaps the most _impactful_ decision, though, is the choice of $f$.
For the scope of this work I'm focusing on NNs - they _theoretically_ can capture all of the performance any other model could and I happen to be more familiar with them than the others.

[^2]: The companion slides go into these in greater detail and with some helpful visualizations.

# Initial Investigation

I've built up a small Python library `nepare` that implements pairwise augmentation strategies and provides a basic NN class that can be used to actually run Neural Pairwise Regression.
The code itself is quite short (this idea is not, ultimately, that complicated) and is currently focused more on flexibility so that the above questions can be explored.

Most of my initial work in benchmarking is saved and visible as Jupyter notebooks.
There are a number of demonstrations in the repository (see [`notebooks`](../../notebooks/)) that show how to fit:

 - `demo`: arbitrary 2D surfaces
 - `polaris_adme`: ADME properties using a fixed molecular representation
 - `polaris_chemprop_nepare`: ADME properties using a learned molecular representation with ChemProp
 - `polaris_asap`: the ASAP discovery challenge with polaris

Each notebook contains extensive inline commentary about the finer points of implementing pairwise regression, but I'll highlight just a few things:

 - the results of `polaris_adme` and `polaris_chemprop_nepare` are ranked _quite_ high on [the RPPB leaderboard](https://polarishub.io/benchmarks/polaris/adme-fang-rppb-1), only narrowly losing out to a massive foundation model
 - the `polaris_asap` notebook shows everything that is needed to reproduce my submission for the ASAP Discovery x OpenADMET Competition's [antiviral-potency](https://polarishub.io/competitions/asap-discovery/antiviral-potency-2025) and [antiviral-admet](https://polarishub.io/competitions/asap-discovery/antiviral-admet-2025) tasks.
 The exact commits showing the run notebook are [here]() for the former and [here](https://github.com/JacksonBurns/neural-pairwise-regression/blob/1a0f3b6b472b3831ab296352925dd77e9aad8d5e/notebooks/polaris_asap/main.ipynb) for the latter.

# Future Directions

Obviously a random spattering of decent results aren't really enough to demonstrate the efficacy of this approach.
First and foremost will be running more and more diverse benchmarks to try and understand the failure modes of this approach.

In addition to the many questions mentioned in the background, there are a host of possible approaches to enforcing loop consistency in the trained model.
This refers to the fact that any closed loop of predicted differences _should_ sum to zero ($\Delta_{1,2} + \Delta_{2,n} + ... + \Delta_{n,1} = 0$).
Whether or not this is possible, or even useful, has a large impact on the way that we would run inference using this framework.

Uncertainty calibration is another important aspect of this approach.
The ensemble of predictions gives an intuitively well-calibrated uncertainty estimate, but this of course needs to be quantified.

Please reach out to me ([GitHub](https://github.com/JacksonBurns/neural-pairwise-regression), jwburns|@|mit.edu, or find my socials [on my personal site](https://www.jacksonburns.us/)) if you are interested in collaborating on any aspect of this!

<!-- 

For posterity, and because I think it's neat to see the refinement process, here are my original random thoughts in no particular order:
## Background
At the most basic level, we want to find some computational model that predicts some target value $y$ based on an input vector $x$ of features (e.g. the value of a house and its square footage and build year, respectively).
This can be done with any model architecture - linear regression, classical machine learning algorithms, and neural networks.
It has shortcoming in that:
 - it cannot (and is not intended to) _extrapolate_ to new feature _or_ target space
 - predictions are made in a vacuum, i.e. there is no idea of how uncertain we are in a prediction
 - all models have parameters ($\theta$), and the more complex, expressive models (like NNs, which are universal approximators) that can tackle the hardest problem have a lot of parameters, thus requiring a lot of data to set them

Changing this problem to instead predict the _difference_ in a target for two inputs attempts to resolve some of these issues:
 - extrapolation _can_ be easier in this formalism, since Out-of-Combination (OOC) prediction _can_ be easier than Out-of-Distribution prediction, see "Learning to Extrapolate" for a theoretical exploration and "Extrapolation is not the same as interpolation" for an emperical example in drug discovery
 - when running inference, one can use all of the training points as 'anchors' and get a (hopefuly) well calibrated prediction of the uncertainty
 - we have way more data 

Order of questions:
Why the interest in this?
Low data regimes, some actual delta applications naturally, ‘it just works’
Why does this work?
OOS vs. OOC, more data is more better (or at least, lets you fit a more complex model)
What architecture?
is technically model agnostic, but NNs are UFAs (important later)
How to represent the input?
Explicit difference? That’s inductive bias a UFA could learn (and could be outright wrong, why would it be manhattan in the feature just because we say so in the target?)
How to build the NN?
Again, avoid inductive bias in setup, NN can learn it (we have so much data!), i.e. don’t need ‘branches’ or ‘operation layers’
How do we train?
Do we want and/or need to enforce self distance of zero (requires training on self-pairs), loop consistency (free?), and 2-loop consistency (requires training on x12 and x21)?
We can attempt to enforce this at the architectural level (satisfy for sure), enforce in loss (satisfy OK), or just check after training (e.g. DeepDelta, probably also OK)
How do we run inference?
Need to use all points? Subset? of most/least/? similar? Weighting of predictions? e.g. smallest diffs are probably similar to training and thus interpolation, trust them more (just geom mean lol). Or, just take the nice uncertainty estimate and run with it?
Choice of training changes what would work here (e.g. could not allow test point to move in the input if we don’t have 2-loop consistency)
How do we scale this?
If this really is just generically more accurate than direct prediction, we certainly want to train everything like this, but now quadratic increase of training data is a burden… how to overcome this? partial augmentation?

RBFE informed 'cycle construction' - programs like LOMAP and HIMAP can create a pathway of small perturbations through a list of species, the idea being that small perturbations are likely to be more accurate in their prediction of the RBFE, giving a more accurate overall prediction. Difficulty here is that we know the exact errors for all of the training deltas but struggle to incorporate those numbers during inference time.
Bayesian Optimization - could select susbsequent anchors to minimize variance

Both of these probably lose out to some simple weighting scheme

can we make the triangle inequality, symmetry, cycle etc. be enforeced in delta learning. check what deepdelta dud - also check if we actually need to enforce it? is it just leraning it on its own? i imagine if we put in the self pairs at least that one will be satisfied

There are additional nice things, like how you can use SHAP [@shap] on this very simple architecture to get an idea of which features for which input are actually contributing to the prediction. 

-->

# Cited Works