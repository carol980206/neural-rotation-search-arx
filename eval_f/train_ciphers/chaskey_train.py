'''
Source code from https://github.com/CryptAnalystDesigner/MutipleCipherDesChaskeyPresent.git
'''
import numpy as np
from os import urandom
from copy import deepcopy


class ChaskeyTrain:
    def __init__(self, c):
        self.WORD_SIZE = 32
        self.MASK_VAL = (2 ** self.WORD_SIZE) - 1
        self.c = c if c is not None else [0] * 6

    def rol(self, x, k):
        k = k % self.WORD_SIZE
        return (((x << k) & self.MASK_VAL) | (x >> (self.WORD_SIZE - k)))

    def ror(self, x, k):
        k = k % self.WORD_SIZE
        return ((x >> k) | ((x << (self.WORD_SIZE - k)) & self.MASK_VAL))

    def permutation_one_round(self, p):
        v = p
        v[0] = (v[0] + v[1]) & self.MASK_VAL
        v[1] = self.rol(v[1], self.c[0])
        v[1] ^= v[0]
        v[0] = self.rol(v[0], self.c[2])
        v[2] = (v[2] + v[3]) & self.MASK_VAL
        v[3] = self.rol(v[3], self.c[1])
        v[3] ^= v[2]
        v[0] = (v[0] + v[3]) & self.MASK_VAL
        v[3] = self.rol(v[3], self.c[4])
        v[3] ^= v[0]
        v[2] = (v[2] + v[1]) & self.MASK_VAL
        v[1] = self.rol(v[1], self.c[3])
        v[1] ^= v[2]
        v[2] = self.rol(v[2], self.c[5])
        for i in range(4):
            v[i] &= self.MASK_VAL
        return v

    def permutation_upper(self, p):
        v = p
        v[0] = (v[0] + v[1]) & self.MASK_VAL
        v[1] = self.rol(v[1], self.c[0])
        v[1] ^= v[0]
        v[0] = self.rol(v[0], self.c[2])
        v[2] = (v[2] + v[3]) & self.MASK_VAL
        v[3] = self.rol(v[3], self.c[1])
        v[3] ^= v[2]
        for i in range(4):
            v[i] &= self.MASK_VAL
        return [v[2], v[1], v[0], v[3]]

    def timesTwo(self, key):
        k0, k1, k2, k3 = key[0], key[1], key[2], key[3]
        tp = deepcopy(k0)
        k0 = (k0 << 1) | ((k1 >> 31) & 1)
        k1 = (k1 << 1) | ((k2 >> 31) & 1)
        k2 = (k2 << 1) | ((k3 >> 31) & 1)
        k3 = k3 << 1
        k3[self.ror(tp, 31) & 1 == 1] = k3[self.ror(tp, 31) & 1 == 1] ^ 0x00000087
        return (k0, k1, k2, k3)

    def subkeys(self, k):
        k1 = self.timesTwo(k)
        k2 = self.timesTwo(k1)
        return (k1, k2)

    def permutation(self, p, k, nr):
        k1, _ = self.subkeys(k)
        s = 4 * [0]
        for i in range(4):
            s[i] = k[i] ^ p[i] ^ k1[i]
        sa, sb, sc, sd = s[0], s[1], s[2], s[3]
        full_rounds = int(nr)
        has_half_round = (nr - full_rounds) >= 0.25
        for _ in range(full_rounds):
            sd, sc, sb, sa = self.permutation_one_round([sd, sc, sb, sa])
        if has_half_round:
            sd, sc, sb, sa = self.permutation_upper([sd, sc, sb, sa])
        sa = sa ^ k1[0]
        sb = sb ^ k1[1]
        sc = sc ^ k1[2]
        sd = sd ^ k1[3]
        return sa, sb, sc, sd

    def convert_to_binary(self, arr):
        X = np.zeros((len(arr) * self.WORD_SIZE, len(arr[0])), dtype=np.uint8)
        for i in range(len(arr) * self.WORD_SIZE):
            index = i // self.WORD_SIZE
            offset = self.WORD_SIZE - (i % self.WORD_SIZE) - 1
            X[i] = (arr[index] >> offset) & 1
        return X.transpose()

    def make_train_data(self, n, nr, diff):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        k = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, n)
        pl0 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        pl1 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        pl2 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        pl3 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        pr0 = pl0 ^ diff[0]
        pr1 = pl1 ^ diff[1]
        pr2 = pl2 ^ diff[2]
        pr3 = pl3 ^ diff[3]
        num_rand_samples = np.sum(Y == 0)
        pr0[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        pr1[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        pr2[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        pr3[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        cl0, cl1, cl2, cl3 = self.permutation((pl0, pl1, pl2, pl3), k, nr)
        cr0, cr1, cr2, cr3 = self.permutation((pr0, pr1, pr2, pr3), k, nr)
        d0 = cl0 ^ cr0
        d1 = cl1 ^ cr1
        d2 = cl2 ^ cr2
        d3 = cl3 ^ cr3
        # C || delta C
        X = [cl0, cl1, cl2, cl3, d0, d1, d2, d3]
        X = self.convert_to_binary(X)
        return (X, Y)
