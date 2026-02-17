import numpy as np
from os import urandom
from copy import deepcopy

class Chaskey(object):
    def __init__(self, c):
        self.WORD_SIZE = 32
        self.BLOCK_SIZE = 128
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
        v[0] = (v[0] + v[1]) & self.MASK_VAL; v[1] = self.rol(v[1], self.c[0]); v[1] ^= v[0]; v[0] = self.rol(v[0], self.c[2]);
        v[2] = (v[2] + v[3]) & self.MASK_VAL; v[3] = self.rol(v[3], self.c[1]); v[3] ^= v[2];
        v[0] = (v[0] + v[3]) & self.MASK_VAL; v[3] = self.rol(v[3], self.c[4]); v[3] ^= v[0];
        v[2] = (v[2] + v[1]) & self.MASK_VAL; v[1] = self.rol(v[1], self.c[3]); v[1] ^= v[2]; v[2] = self.rol(v[2], self.c[5]);

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
    
    def permutation_lower(self, p):
        v = p
        v[0] = (v[2] + v[3]) & self.MASK_VAL
        v[3] = self.rol(v[3], self.c[4])
        v[3] ^= v[0]
        v[2] = (v[0] + v[1]) & self.MASK_VAL
        v[1] = self.rol(v[1], self.c[3])
        v[1] ^= v[2]
        v[2] = self.rol(v[2], self.c[5])

        for i in range(4):
            v[i] &= self.MASK_VAL

        return v
        
    def timesTwo(self, key):
        k0, k1, k2, k3 = key[0], key[1], key[2], key[3]  # k3[0] -> lsb, k0[31] -> msb
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
    
    def expand_key(self, mk, nr, key_bit_length=128):
        k = mk
        k1, k2 = self.subkeys(k)
        return {
            "k": k,
            "k1": k1,
            "k2": k2,
            "nr": nr
        }

    def encrypt(self, p, rks):
        k, k1, _, nr = rks["k"], rks["k1"], rks["k2"], rks["nr"]
        half_r_lower, half_r_upper = False, False
        # Determine whether nr contains a half-round
        if not isinstance(nr, int):
            assert (nr - int(nr)) == 0.5, "Chaskey round num error!"
            half_r_upper = True
            half_r_lower = False
        s = 4 * [0]
        for i in range(4):
            s[i] = k[i] ^ p[i] ^ k1[i]  # k[3][0] -> lsb, k[0][31] -> msb
        sa, sb, sc, sd = s[0], s[1], s[2], s[3]
        if half_r_lower == True:
            sd, sc, sb, sa = self.permutation_lower([sd, sc, sb, sa])
        for _ in range(int(nr)):
            sd, sc, sb, sa = self.permutation_one_round([sd, sc, sb, sa])
        if half_r_upper == True:
            sd, sc, sb, sa = self.permutation_upper([sd, sc, sb, sa])
        sa, sb, sc, sd = sa ^ k1[0], sb ^ k1[1], sc ^ k1[2], sd ^ k1[3]
        return sa, sb, sc, sd
    
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
        k = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, n)
        pl0 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        pl1 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        pl2 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        pl3 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        pr0, pr1, pr2, pr3 = pl0 ^ diff[0], pl1 ^ diff[1], pl2 ^ diff[2], pl3 ^ diff[3]
        num_rand_samples = np.sum(Y == 0)
        pr0[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        pr1[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        pr2[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        pr3[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        rks = self.expand_key(k, nr)
        cl0, cl1, cl2, cl3 = self.encrypt((pl0, pl1, pl2, pl3), rks)
        cr0, cr1, cr2, cr3 = self.encrypt((pr0, pr1, pr2, pr3), rks)
        
        cl2 = self.ror((self.ror(cl1, self.c[2]) ^ cl2), self.c[0])
        cl1 = self.ror(cl1, self.c[2])
        cl0 = self.ror((cl3 ^ cl0), self.c[1])
        
        cr2 = self.ror((self.ror(cr1, self.c[2]) ^ cr2), self.c[0])
        cr1 = self.ror(cr1, self.c[2])
        cr0 = self.ror((cr3 ^ cr0), self.c[1])
        
        if data_form == 'only_diff':
            d0, d1, d2, d3 = cl0 ^ cr0, cl1 ^ cr1, cl2 ^ cr2, cl3 ^ cr3
            X = [d0, d1, d2, d3]
        X = self.convert_to_binary(X)
        return (X, Y)

    def make_train_data_multidiff(self, n, nr, diffs, data_form='only_diff'):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        k = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, n)
        p = [np.frombuffer(urandom(4 * n), dtype=np.uint32) for _ in range(4)]
        paired_p = []
        for diff in diffs:
            paired_p.append([p[i] ^ diff[i] for i in range(4)])
        negative_pos = (Y == 0)
        num_rand_samples = np.sum(negative_pos)
        for i in range(len(diffs)):
            for j in range(4):
                paired_p[i][j][negative_pos] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        rks = self.expand_key(k, nr)
        c = self.encrypt(p, rks)
        paired_c = [self.encrypt(x, rks) for x in paired_p]
        if data_form == 'only_diff':
            diff_words = []
            for x in paired_c:
                diff_words += [c[i] ^ x[i] for i in range(4)]
            X = diff_words
        X = self.convert_to_binary(X)
        return (X, Y)