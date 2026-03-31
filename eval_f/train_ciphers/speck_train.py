'''
Source code from https://github.com/agohr/deep_speck.git
'''
import numpy as np
from os import urandom


class SpeckTrain:
    CONFIGS = {
        'speck32':  {'word_size': 16, 'dtype': np.uint16, 'byte_size': 2, 'key_version': 64},
        'speck48':  {'word_size': 24, 'dtype': np.uint32, 'byte_size': 4, 'key_version': 72},
        'speck64':  {'word_size': 32, 'dtype': np.uint32, 'byte_size': 4, 'key_version': 96},
        'speck96':  {'word_size': 48, 'dtype': np.uint64, 'byte_size': 8, 'key_version': 144},
        'speck128': {'word_size': 64, 'dtype': np.uint64, 'byte_size': 8, 'key_version': 128},
    }

    def __init__(self, algorithm, alpha, beta):
        config = self.CONFIGS[algorithm]
        self.alpha = alpha
        self.beta = beta
        self.WORD_SIZE = config['word_size']
        self.MASK_VAL = (2 ** self.WORD_SIZE) - 1
        self.dtype = config['dtype']
        self.byte_size = config['byte_size']
        self.key_version = config['key_version']
        self.needs_mask = (self.WORD_SIZE < self.dtype(0).itemsize * 8)

    def rol(self, x, k):
        k = k % self.WORD_SIZE
        return (((x << k) & self.MASK_VAL) | (x >> (self.WORD_SIZE - k)))

    def ror(self, x, k):
        k = k % self.WORD_SIZE
        return ((x >> k) | ((x << (self.WORD_SIZE - k)) & self.MASK_VAL))

    def enc_one_round(self, p, k):
        c0, c1 = p[0], p[1]
        c0 = self.ror(c0, self.alpha)
        c0 = (c0 + c1) & self.MASK_VAL
        c0 = c0 ^ k
        c1 = self.rol(c1, self.beta)
        c1 = c1 ^ c0
        return (c0, c1)

    def expand_key(self, k, t):
        version = self.key_version
        ks = [0 for _ in range(t)]
        ks[0] = k[len(k) - 1]
        l = list(reversed(k[:len(k) - 1]))
        ver = version // self.WORD_SIZE
        for i in range(t - 1):
            l[i % (ver - 1)], ks[i + 1] = self.enc_one_round((l[i % (ver - 1)], ks[i]), i)
        return ks

    def encrypt(self, p, ks):
        x, y = p[0], p[1]
        for k in ks:
            x, y = self.enc_one_round((x, y), k)
        return (x, y)

    def convert_to_binary(self, arr):
        n = len(arr[0])
        X = np.zeros((len(arr) * self.WORD_SIZE, n), dtype=np.uint8)
        for i in range(len(arr) * self.WORD_SIZE):
            index = i // self.WORD_SIZE
            offset = self.WORD_SIZE - (i % self.WORD_SIZE) - 1
            X[i] = (arr[index] >> offset) & 1
        return X.transpose()

    def _random_array(self, n):
        arr = np.frombuffer(urandom(self.byte_size * n), dtype=self.dtype)
        if self.needs_mask:
            arr = arr & self.MASK_VAL
        return arr

    def _random_key(self, n):
        mk_num = self.key_version // self.WORD_SIZE
        mk = np.frombuffer(urandom(self.byte_size * n * mk_num), dtype=self.dtype).reshape((mk_num, n))
        if self.needs_mask:
            mk = mk & self.MASK_VAL
        return mk

    def make_train_data(self, n, nr, diff):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        mk = self._random_key(n)
        rk = self.expand_key(mk, nr)
        p0l = self._random_array(n)
        p0r = self._random_array(n)
        p1l = p0l ^ diff[0]
        p1r = p0r ^ diff[1]
        rand_pos = (Y == 0)
        num_rand = np.sum(rand_pos)
        p1l[rand_pos] = self._random_array(num_rand)
        p1r[rand_pos] = self._random_array(num_rand)
        c0l, c0r = self.encrypt((p0l, p0r), rk)
        c1l, c1r = self.encrypt((p1l, p1r), rk)
        # Calculating back
        c0r = self.ror((c0l ^ c0r), self.beta)
        c1r = self.ror((c1l ^ c1r), self.beta)
        # Ciphertext pair
        X = [c0l, c0r, c1l, c1r]
        X = self.convert_to_binary(X)
        return (X, Y)
