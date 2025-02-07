---
title: Neural Pairwise Regression
author: Jackson Burns
header-includes: |
    \usetheme{Custom}
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

## Schedule

\tableofcontents

# Formatting Examples

## Parallel Columns

<!-- :::::::::::::: {.columns}
::: {.column width="40%"}
Left Column Content
:::
::: {.column width="60%"}
 - Right column content
   - sub-content
:::
:::::::::::::: -->

::: columns

:::: column
left column content
::::

:::: column
 - right column content
::::

:::

## Table

| Customer / Source | Arnhem [euro/ton] |  Gouda [euro/ton] | Demand [tons] |
| :---: | :--: | :--: | :--: |
| London | n/a | 2.5 | 125 |
| The Hague | 1.4 | 0.8 | 200 |
| **Supply [tons]** | 550 tons | 700 tons | n/a |

## Citations and Math

 - We can cite papers inline [@shap] using `[@name]`, based on the contents of `refs.bib`
 - Math will be rendered nicely, i.e. $y=f(x;\theta)$ and $y_1-y_2=f(x_1,x_2;\theta)$

# Further Examples

## Figures, References, Title-less Slides

(Figure \ref{logo}) - an existing approach formalized into an easy-to-use, reliable, and correct implementation.
`fastprop` is highly modular for seamless integration into existing workflows and includes and end-to-end Command Line Interface (CLI) for general use.

##

![`fastprop` logo.\label{logo}](figures/dogs.jpg){ width=2in }

## Formatted Code

We can show code beautifully like this:
```python
from py2opsin import py2opsin

smiles_string = py2opsin(
  chemical_name = "ethane",
  output_format = "SMILES",
)
```

# Supplementary Material

## Obnoxiously Long Equations

And so and and so forth

## Cited Works