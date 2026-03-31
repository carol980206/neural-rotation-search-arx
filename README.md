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
- An exact evaluation function `f` based on evolutionary input-difference search and neural distinguisher training (Source code from https://github.com/Crypto-TII/AutoND.git)

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

├── eval_f/                          # Exact evaluation function f

│   ├── main.py                      # Entry point for eval_f

│   ├── search.py                    # Evolutionary input-difference search interface

│   ├── optimizer.py                 # Evolutionary algorithm and bias-score optimizer

│   ├── trainer.py                   # Neural distinguisher training pipeline

│   ├── ciphers/                     # Vectorized cipher implementations for search

│   └── train_ciphers/               # Cipher implementations for ND training data generation

├── eval_f_hat/                      # Approximate evaluation function f_hat

│   ├── search.py                    # Main entry point for parameter search

│   ├── param_search_algorithm.py    # Greedy-with-exploration parameter search algorithm

│   ├── neural_evaluator.py          # f_hat implementations: ND and JND training pipelines

│   ├── model.py                     # Residual neural network architecture definition

│   ├── detect_important_diff.py     # Difference sensitivity analysis

│   ├── eval_diff_score.py           # Compute bias scores for input differences

│   ├── eval_func.py                 # Evaluation wrapper with caching and statistics tracking

│   ├── util.py                      # Utility and helper functions

│   └── ciphers/                     # Implementations of ARX-based ciphers

└── README.md

## Evaluate a Given Parameter with `eval_f`

Given a specific rotation parameter, `eval_f` performs:
1. Evolutionary search for good input differences (Source code from https://github.com/Crypto-TII/AutoND.git).
2. (Optional) Training a neural distinguisher on each found difference.

Example:

```bash
# Search input differences only
cd eval_f
python main.py speck32 7 "[[7,2]]"

# Search and train neural distinguishers
python main.py speck32 7 "[[7,2]]" --train

# Chaskey uses half-round notation
python main.py chaskey 3.5 "[[5,8,16,7,13,16]]" --train

# Multiple parameter sets with parallel search
python main.py speck64 7 "[[8,3],[7,2]]" --cores 2 --train
```

Arguments:
- `algorithm`: speck32, speck48, speck64, speck96, speck128, lea, chaskey, siphash
- `max_round`: number of cipher rounds (e.g. 7 for speck, 3.5 for chaskey)
- `params`: rotation parameter sets as JSON
- `--train`: enable neural distinguisher training after search
- `--output`: results directory (default: results)
- `--epsilon`: epsilon for close differences (default: 0.1)
- `--cores`: CPU cores for parallel search (default: 1)
- `--max-diffs`: max differences to train on (default: 4)

## Search for Secure Rotation Parameters with `eval_f_hat`

Example:

```bash
cd eval_f_hat
python search.py speck128
python search.py lea
python search.py chaskey
python search.py siphash
```
