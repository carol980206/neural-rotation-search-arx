'''
Source code from https://github.com/CryptAnalystDesigner/MutipleCipherDesChaskeyPresent.git
'''
import numpy as np
from copy import deepcopy

plain_bits = 128
key_bits = 128
word_size = 32

def WORD_SIZE():
    return(32)

MASK_VAL = 2 ** WORD_SIZE() - 1

def rol(x, k):
    k = k % WORD_SIZE()
    return (((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))

def ror(x, k):
    k = k % WORD_SIZE()
    return ((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))

def permutation_one_round(p, c):
    v = p
    v[0] = (v[0] + v[1]) & MASK_VAL; v[1] = rol(v[1], c[0]); v[1] ^= v[0]; v[0] = rol(v[0], c[2]);
    v[2] = (v[2] + v[3]) & MASK_VAL; v[3] = rol(v[3], c[1]); v[3] ^= v[2];
    v[0] = (v[0] + v[3]) & MASK_VAL; v[3] = rol(v[3], c[4]); v[3] ^= v[0];
    v[2] = (v[2] + v[1]) & MASK_VAL; v[1] = rol(v[1], c[3]); v[1] ^= v[2]; v[2] = rol(v[2], c[5]);

    for i in range(4):
        v[i] &= MASK_VAL

    return v

def permutation_upper(p, c):
    v = p
    v[0] = (v[0] + v[1]) & MASK_VAL
    v[1] = rol(v[1], c[0])
    v[1] ^= v[0]
    v[0] = rol(v[0], c[2])
    v[2] = (v[2] + v[3]) & MASK_VAL
    v[3] = rol(v[3], c[1])
    v[3] ^= v[2]

    for i in range(4):
        v[i] &= MASK_VAL

    return [v[2], v[1], v[0], v[3]]

def permutation_lower(p, c):
    v = p
    v[0] = (v[2] + v[3]) & MASK_VAL
    v[3] = rol(v[3], c[4])
    v[3] ^= v[0]
    v[2] = (v[0] + v[1]) & MASK_VAL
    v[1] = rol(v[1], c[3])
    v[1] ^= v[2]
    v[2] = rol(v[2], c[5])

    for i in range(4):
        v[i] &= MASK_VAL

    return v

def timesTwo(key):
    k0, k1, k2, k3 = key[0], key[1], key[2], key[3]
    tp = deepcopy(k0)
    k0 = (k0 << 1) | ((k1 >> 31) & 1)
    k1 = (k1 << 1) | ((k2 >> 31) & 1)
    k2 = (k2 << 1) | ((k3 >> 31) & 1)
    k3 = k3 << 1

    k3[ror(tp, 31) & 1 == 1] = k3[ror(tp, 31) & 1 == 1] ^ 0x00000087
    return (k0, k1, k2, k3)

def subkeys(k):
    k1 = timesTwo(k)
    k2 = timesTwo(k1)
    return (k1, k2)

def encrypt(P, K, nr, c):
    p = convert_from_binary(P).transpose()
    k = convert_from_binary(K).transpose()
    k1, _ = subkeys(k)
    full_rounds = int(nr)
    has_half_round = (nr - full_rounds) >= 0.25  # True for x.5 values
    s = 4 * [0]
    for i in range(4):
        s[i] = k[i] ^ p[i] ^ k1[i]
    sa, sb, sc, sd = s[0], s[1], s[2], s[3]
    for _ in range(full_rounds):
        sd, sc, sb, sa = permutation_one_round([sd, sc, sb, sa], c)
    if has_half_round:
        sd, sc, sb, sa = permutation_upper([sd, sc, sb, sa], c)
    sa, sb, sc, sd = sa ^ k1[0], sb ^ k1[1], sc ^ k1[2], sd ^ k1[3]
    X = convert_to_binary([sa, sb, sc, sd])
    return X

def convert_to_binary(arr):
    X = np.zeros((len(arr) * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
    for i in range(len(arr) * WORD_SIZE()):
        index = i // WORD_SIZE();
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
        X[i] = (arr[index] >> offset) & 1;
    X = X.transpose();
    return(X);

def convert_from_binary(arr, _dtype=np.uint32):
    num_words = arr.shape[1] // WORD_SIZE()
    X = np.zeros((len(arr), num_words), dtype=_dtype)
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE() * i + j
            X[:, i] += 2 ** (WORD_SIZE() - 1 - j) * arr[:, pos]
    return X
