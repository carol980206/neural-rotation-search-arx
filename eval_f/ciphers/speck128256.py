import numpy as np

plain_bits = 128
key_bits = 256
word_size = 64

def WORD_SIZE():
    return(64);

MASK_VAL = 2 ** WORD_SIZE() - 1;

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def enc_one_round(p, k, param):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, param[0]);
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, param[1]);
    c1 = c1 ^ c0;
    return(c0,c1);

def expand_key(k, t, param):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i, param);
    return(ks);

def encrypt(p, k, r, param):
    P = convert_from_binary(p)
    K = convert_from_binary(k).transpose()
    ks = expand_key(K, r, param)
    x, y = P[:, 0], P[:, 1];
    for i in range(r):
        rk = ks[i]
        x,y = enc_one_round((x,y), rk, param);
    return convert_to_binary([x, y]);

def convert_to_binary(arr):
    X = np.zeros((len(arr) * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
    for i in range(len(arr) * WORD_SIZE()):
        index = i // WORD_SIZE();
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
        X[i] = (arr[index] >> offset) & 1;
    X = X.transpose();
    return(X);

def convert_from_binary(arr, _dtype=np.uint64):
    num_words = arr.shape[1]//WORD_SIZE()
    X = np.zeros((len(arr), num_words),dtype=_dtype);
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE()*i+j
            X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]
    return(X);
