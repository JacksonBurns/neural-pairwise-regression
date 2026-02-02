# Neural Pairwise Regression
Recasting the archetypal neural network regression problem $y=f(x;\theta)$ to $y_1-y_2=f(x_1,x_2;\theta)$ for improved extrapolation, well-calibrated uncertainty estimates, and effectiveness in low-data regimes.

# Getting Started
This repository is laid out as follows:
 - `meta` contains a paper and presentation presenting the theory of Neural Pairwise Regression, helpful for getting started.
 - `nepare` has the actual source code implementing the Neural Pairwise Regression algorithm as a python package (also called `nepare`) which can be installed by running `pip install .` from the current directory.
 - `notebooks` contains Jupyter notebooks demonstrating how to use this code actual regression problems.

# Future Directions

Once a neural pairwise regressor is trained, it's unclear how to use it for inference.
How many/which 'anchors' to use - maybe just all, to get well-calibrated (?) uncertainty based on the ensemble's stdev?
Is there an optimal choice?

The latter is more interesting, and papers like [Optimal Measurement Network of Pairwise Differences](https://doi.org/10.1021/acs.jcim.9b00528) lay out a way to find 'closing loops' (see also their [source code](https://github.com/forcefield/DiffNet)).
