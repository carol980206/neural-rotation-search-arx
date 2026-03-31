import numpy as np
from os import urandom


class SiphashTrain:
    def __init__(self, c):
        self.WORD_SIZE = 64
        self.MASK_VAL = (2 ** self.WORD_SIZE) - 1
        self.c = c if c is not None else [0] * 6

    def rol(self, x, k):
        k = k % self.WORD_SIZE
        return (((x << k) & self.MASK_VAL) | (x >> (self.WORD_SIZE - k)))

    def permutation_one_round(self, p):
        v = p
        v[0] = (v[0] + v[1]) & self.MASK_VAL
        v[1] = self.rol(v[1], self.c[0])
        v[1] = v[1] ^ v[0]
        v[2] = (v[2] + v[3]) & self.MASK_VAL
        v[3] = self.rol(v[3], self.c[1])
        v[3] = v[3] ^ v[2]
        v[0] = self.rol(v[0], self.c[2])
        v[2] = (v[2] + v[1]) & self.MASK_VAL
        v[1] = self.rol(v[1], self.c[3])
        v[1] = v[1] ^ v[2]
        v[0] = (v[0] + v[3]) & self.MASK_VAL
        v[3] = self.rol(v[3], self.c[4])
        v[3] = v[3] ^ v[0]
        v[2] = self.rol(v[2], self.c[5])
        for i in range(4):
            v[i] = v[i] & self.MASK_VAL
        return v

    def permutation_half_round(self, p):
        v = p
        v[0] = (v[0] + v[1]) & self.MASK_VAL
        v[1] = self.rol(v[1], self.c[0])
        v[1] = v[1] ^ v[0]
        v[2] = (v[2] + v[3]) & self.MASK_VAL
        v[3] = self.rol(v[3], self.c[1])
        v[3] = v[3] ^ v[2]
        v[0] = self.rol(v[0], self.c[2])
        return [v[2], v[1], v[0], v[3]]

    def encrypt(self, p, k0, k1, nr):
        half_r = False
        if not isinstance(nr, int):
            if (nr - int(nr)) >= 0.25:
                half_r = True
        nr_int = int(nr)
        c = 2
        d = nr_int - 2
        v = [0x736f6d6570736575, 0x646f72616e646f6d,
             0x6c7967656e657261, 0x7465646279746573]
        v[0] = v[0] ^ k0
        v[1] = v[1] ^ k1
        v[2] = v[2] ^ k0
        v[3] = v[3] ^ k1 ^ p
        for _ in range(c):
            v[0], v[1], v[2], v[3] = self.permutation_one_round([v[0], v[1], v[2], v[3]])
        v[0] = v[0] ^ p
        v[2] = v[2] ^ 0xff
        for _ in range(d):
            v[0], v[1], v[2], v[3] = self.permutation_one_round([v[0], v[1], v[2], v[3]])
        if half_r:
            v[0], v[1], v[2], v[3] = self.permutation_half_round([v[0], v[1], v[2], v[3]])
        return v[0] ^ v[1] ^ v[2] ^ v[3]

    def convert_to_binary(self, arr):
        X = np.zeros((len(arr) * self.WORD_SIZE, len(arr[0])), dtype=np.uint8)
        for i in range(len(arr) * self.WORD_SIZE):
            index = i // self.WORD_SIZE
            offset = self.WORD_SIZE - (i % self.WORD_SIZE) - 1
            X[i] = (arr[index] >> offset) & 1
        return X.transpose()

    def make_train_data(self, n, nr, diff):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        k0 = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        k1 = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        pl = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        pr = pl ^ diff
        num_rand_samples = np.sum(Y == 0)
        pr[Y == 0] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
        cl = self.encrypt(pl, k0, k1, nr)
        cr = self.encrypt(pr, k0, k1, nr)
        # C || delta C
        X = [cl, cl ^ cr]
        X = self.convert_to_binary(X)
        return (X, Y)
