import numpy as np

plain_bits = 64
key_bits = 64
word_size = 64

def WORD_SIZE():
    return(64)

MASK_VAL = 2 ** WORD_SIZE() - 1

def rol(x, k):
    k = k % WORD_SIZE()
    return (((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))

def permutation_one_round(p, c):
    v = p

    v[0] = (v[0] + v[1]) & MASK_VAL
    v[1] = rol(v[1], c[0])
    v[1] = v[1] ^ v[0]
    v[2] = (v[2] + v[3]) & MASK_VAL
    v[3] = rol(v[3], c[1])
    v[3] = v[3] ^ v[2]
    v[0] = rol(v[0], c[2])

    v[2] = (v[2] + v[1]) & MASK_VAL
    v[1] = rol(v[1], c[3])
    v[1] = v[1] ^ v[2]
    v[0] = (v[0] + v[3]) & MASK_VAL
    v[3] = rol(v[3], c[4])
    v[3] = v[3] ^ v[0]
    v[2] = rol(v[2], c[5])

    for i in range(4):
        v[i] = v[i] & MASK_VAL

    return v

def permutation_half_round(p, c):
    v = p

    v[0] = (v[0] + v[1]) & MASK_VAL
    v[1] = rol(v[1], c[0])
    v[1] = v[1] ^ v[0]
    v[2] = (v[2] + v[3]) & MASK_VAL
    v[3] = rol(v[3], c[1])
    v[3] = v[3] ^ v[2]
    v[0] = rol(v[0], c[2])

    return [v[2], v[1], v[0], v[3]]

def encrypt(P, K0, K1, nr, con):
    p = convert_from_binary(P).squeeze()
    k0 = convert_from_binary(K0).squeeze()
    k1 = convert_from_binary(K1).squeeze()
    # Check whether nr contains a half round
    half_r = False
    if not isinstance(nr, int):
        assert (nr - int(nr)) == 0.5, "SipHash round num error!"
        half_r = True
    nr = int(nr)
    if 1 == nr:
        c = 1; d = 0
    else:
        c = 2; d = nr - 2  # SipHash-2-x
    v = [0x736f6d6570736575, 0x646f72616e646f6d, 0x6c7967656e657261, 0x7465646279746573]
    sa, sb, sc, sd = v[0] ^ k0, v[1] ^ k1, v[2] ^ k0, v[3] ^ k1 ^ p
    for _ in range(c):
        sa, sb, sc, sd = permutation_one_round([sa, sb, sc, sd], con)
    sa = sa ^ p
    sc = sc ^ 0xff
    for _ in range(d):
        sa, sb, sc, sd = permutation_one_round([sa, sb, sc, sd], con)
    if half_r:
        sa, sb, sc, sd = permutation_half_round([sa, sb, sc, sd], con)
    x = sa ^ sb ^ sc ^ sd
    X = convert_to_binary([x])
    return X

def convert_to_binary(arr):
    X = np.zeros((len(arr) * WORD_SIZE(), len(arr[0])), dtype=np.uint8);
    for i in range(len(arr) * WORD_SIZE()):
        index = i // WORD_SIZE();
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
        X[i] = (arr[index] >> offset) & 1;
    X = X.transpose();
    return(X);

def convert_from_binary(arr, _dtype=np.uint64):
    num_words = arr.shape[1] // WORD_SIZE()
    X = np.zeros((len(arr), num_words), dtype=_dtype)
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE() * i + j
            X[:, i] += 2 ** (WORD_SIZE() - 1 - j) * arr[:, pos]
    return X
