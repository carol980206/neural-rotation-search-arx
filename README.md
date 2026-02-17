# Neural-Assisted Rotation Parameter Evaluation and Search for ARX Ciphers

## Overview

This repository provides the implementation of the neural-network-based framework 
proposed in our paper (under submission):

**Neural-Assisted Evaluation and Selection of Rotation Parameters in ARX Ciphers**

The framework evaluates and searches rotation parameters in ARX ciphers 
from the perspective of differential-neural cryptanalysis.

We introduce:

- An efficient approximate evaluation function `f_hat`
- A greedy-with-exploration optimizer for large parameter spaces
- A joint neural distinguisher ($JND$) with difference sensitivity test

The framework is validated on:

- SPECK (32/48/64/96/128)
- LEA
- Chaskey
- SipHash

## Methodology

For a rotation parameter choice $θ$ and $r$-round cipher $E^r_θ$:

1. AutoND-style bias score is computed on $r-1$ rounds.
2. Top-T candidate differences are selected.
3. A Joint Neural Distinguisher ($JND$) is trained.
4. Difference Sensitivity Test identifies the best $Δ^*$.
5. A neural distinguisher $ND(E^r_θ, Δ^*)$ is trained.
6. Evaluation score is derived from classification accuracy.

## Repository Structure

├── ciphers/                     # Implementations of ARX-based ciphers

├── detect_important_diff.py     # Difference sensitivity analysis

├── eval_diff_score.py           # Compute bias scores for input differences

├── eval_func.py                 # Evaluation wrapper with caching and statistics tracking

├── model.py                     # Residual neural network architecture definition

├── neural_evaluator.py          # f_hat implementations: $\mathcal{ND}$ and $\mathcal{JND}$ training pipelines

├── param_search_algorithm.py    # Greedy-with-exploration parameter search algorithm

├── utils.py                     # Utility and helper functions

└── search.py                    # Main entry point for parameter search

## Search for Secure Rotation Parameters

Example:

- python search.py speck128
- python search.py lea
- python search.py chaskey
- python search.py siphash
