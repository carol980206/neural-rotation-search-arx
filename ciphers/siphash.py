import numpy as np
from os import urandom
from copy import deepcopy

class Siphash(object):
    def __init__(self, c):
        self.WORD_SIZE = 64
        self.BLOCK_SIZE = 256
        self.MASK_VAL = (2**self.WORD_SIZE) - 1
        if c is None:
            self.c = 6 * [0]
        else:
            self.c = c  # 6 constants

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
    
    def expand_key(self, mk, nr, key_bit_length=128):
        k0 = mk[0]
        k1 = mk[1]
        return {
            "k0": k0,
            "k1": k1,
            "nr": nr
        }

    def encrypt(self, p, rks):
        k0, k1, nr = rks["k0"], rks["k1"], rks["nr"]
        # Determine whether nr contains a half-round
        half_r = False
        if not isinstance(nr, int):
            assert (nr - int(nr)) == 0.5, "SipHash round num error!"
            half_r = True
        nr = int(nr)
        # SipHash-2-x
        c = 2
        d = nr - 2
        
        v = [0x736f6d6570736575, 0x646f72616e646f6d, 0x6c7967656e657261, 0x7465646279746573]
        v[0], v[1], v[2], v[3] = v[0] ^ k0, v[1] ^ k1, v[2] ^ k0, v[3] ^ k1 ^ p
        for _ in range(c):
            v[0], v[1], v[2], v[3] = self.permutation_one_round([v[0], v[1], v[2], v[3]])
        v[0] = v[0] ^ p
        v[2] = v[2] ^ 0xff
        for _ in range(d):
            v[0], v[1], v[2], v[3] = self.permutation_one_round([v[0], v[1], v[2], v[3]])
        if True == half_r:
            v[0], v[1], v[2], v[3] = self.permutation_half_round([v[0], v[1], v[2], v[3]])
        
        return v[0] ^ v[1] ^ v[2] ^ v[3]
    
    def convert_to_binary(self, arr):
        X = np.zeros((len(arr) * self.WORD_SIZE, len(arr[0])), dtype=np.uint8)
        for i in range(len(arr) * self.WORD_SIZE):
            index = i // self.WORD_SIZE
            offset = self.WORD_SIZE - (i % self.WORD_SIZE) - 1
            X[i] = (arr[index] >> offset) & 1
        X = X.transpose()
        return X

    def make_train_data(self, n, nr, diff, data_form='only_diff', master_key_bit_length=128):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        k = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        k1 = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        pl = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        pr = pl ^ diff[0]
        num_rand_samples = np.sum(Y == 0)
        pr[Y == 0] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
        rks = self.expand_key(k, nr)
        rks["k0"], rks["k1"] = k, k1

        cl = self.encrypt(pl, rks)
        cr = self.encrypt(pr, rks)
        if data_form == 'only_diff':
            d = cl ^ cr
            X = [d]
        X = self.convert_to_binary(X)
        return (X, Y)

    def make_train_data_multidiff(self, n, nr, diffs, data_form='only_diff'):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        k = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        k1 = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        p = [np.frombuffer(urandom(8 * n), dtype=np.uint64)]
        paired_p = []
        for diff in diffs:
            paired_p.append([p[i] ^ diff[i]])
        negative_pos = (Y == 0)
        num_rand_samples = np.sum(negative_pos)
        for i in range(len(diffs)):
            paired_p[i][0][negative_pos] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
        rks = self.expand_key(k, nr)
        rks["k0"], rks["k1"] = k, k1
        c = self.encrypt(p, rks)
        paired_c = [self.encrypt(x, rks) for x in paired_p]
        if data_form == 'only_diff':
            diff_words = []
            for x in paired_c:
                diff_words += [c[i] ^ x[i] for i in range(1)]
            X = diff_words
        X = self.convert_to_binary(X)
        return (X, Y)
    
    @classmethod
    def check_test_vector(cls):
        cipher = cls([13, 16, 32, 17, 21, 32])
        nr = 0
        plaintext = 0x0706050403020100
        key = 0x0f0e0d0c0b0a09080706050403020100
        ciphertext = 0x5e6241c1949a142a
        rks = cipher.expand_key(key, nr)
        c = cipher.encrypt(plaintext, rks)
        if c == ciphertext:
            print('Test vector is verified!')
        else:
            print('Test vector is not verified!')

if __name__ == '__main__':
    Siphash.check_test_vector()