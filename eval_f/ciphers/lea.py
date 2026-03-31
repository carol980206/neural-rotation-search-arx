import numpy as np

plain_bits = 128
key_bits = 128
word_size = 32

def WORD_SIZE():
    return(32);

MASK_VAL = 2 ** WORD_SIZE() - 1;
DELTA=np.array([0xc3efe9db,0x44626b02,0x79e27c8a,0x78df30ec,0x715ea49e,0xc785da0a,0xe04ef22a,0xe5c40957])

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def expand_key(K, t):
    ks = [0 for i in range(t)];
    tmp = [x for x in K]
    for i in range(t):
        tmp[0]=rol(    (tmp[0]+rol(DELTA[i%4], i) ) & MASK_VAL,1)
        tmp[1]=rol(    (tmp[1]+rol(DELTA[i%4], i+1) ) & MASK_VAL,3)
        tmp[2]=rol(    (tmp[2]+rol(DELTA[i%4], i+2) ) & MASK_VAL,6)
        tmp[3]=rol(    (tmp[3]+rol(DELTA[i%4], i+3) ) & MASK_VAL,11)
        ks[i]=np.array([tmp[0],tmp[1],tmp[2],tmp[1],tmp[3],tmp[1]])
    return(np.array(ks));

def encrypt(p, k, r, c):
    P = convert_from_binary(p).byteswap().transpose()
    K = convert_from_binary(k).byteswap().transpose()
    ks = expand_key(K, r)
    for i in range(r):
        p0, p1, p2, p3 = P.copy()
        k0, k1, k2, k3, k4, k5 = ks[i]
        P[3] = p0
        P[0] = rol(((p0^k0)+(p1^k1)) & MASK_VAL, c[0])
        P[1] = ror(((p1^k2)+(p2^k3)) & MASK_VAL, c[1])
        P[2] = ror(((p2^k4)+(p3^k5)) & MASK_VAL, c[2])
    return(convert_to_binary(P.byteswap()));

def convert_to_binary(arr):
    X = np.zeros((len(arr) * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
    for i in range(len(arr) * WORD_SIZE()):
        index = i // WORD_SIZE();
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
        X[i] = (arr[index] >> offset) & 1;
    X = X.transpose();
    return(X);

def convert_from_binary(arr, _dtype=np.uint32):
    num_words = arr.shape[1]//WORD_SIZE()
    X = np.zeros((len(arr), num_words),dtype=_dtype);
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE()*i+j
            X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]
    return(X);
