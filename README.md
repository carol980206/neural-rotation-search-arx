# Neural-Assisted Rotation Parameter Evaluation and Search for ARX Ciphers

## Overview

This repository provides the implementation of the neural-network-based framework 
proposed in our paper:

**Neural-Assisted Evaluation and Selection of Rotation Parameters in ARX Ciphers**

The framework evaluates and searches rotation parameters in ARX ciphers 
from the perspective of differential-neural cryptanalysis.

We introduce:

- An efficient approximate evaluation function `f_hat`
- A greedy-with-exploration optimizer for large parameter spaces
- A joint neural distinguisher (JND) with difference sensitivity test

The framework is validated on:

- SPECK (32/48/64/96/128)
- LEA
- Chaskey
- SipHash


## Search for Secure Rotation Parameters

Example:

- python search.py speck128
- python search.py lea
- python search.py chaskey
