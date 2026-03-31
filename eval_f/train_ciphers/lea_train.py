import numpy as np
from os import urandom

delta = [0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec,
         0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957]
delta = np.array(delta, dtype=np.uint32).reshape(8, 1)


class LEATrain:
    def __init__(self, a, b, c):
        self.WORD_SIZE = 32
        self.MASK_VAL = (2 ** self.WORD_SIZE) - 1
        self.a = a
        self.b = b
        self.c = c

    def rol(self, x, k):
        k = k % self.WORD_SIZE
        return (((x << k) & self.MASK_VAL) | (x >> (self.WORD_SIZE - k)))

    def ror(self, x, k):
        k = k % self.WORD_SIZE
        return ((x >> k) | ((x << (self.WORD_SIZE - k)) & self.MASK_VAL))

    def enc_one_round(self, p, rk):
        c0 = self.rol((p[0] ^ rk[0]) + (p[1] ^ rk[1]), self.a)
        c1 = self.ror((p[1] ^ rk[2]) + (p[2] ^ rk[3]), self.b)
        c2 = self.ror((p[2] ^ rk[4]) + (p[3] ^ rk[5]), self.c)
        c3 = p[0].copy()
        return (c0, c1, c2, c3)

    def expand_key(self, mk, nr, key_bit_length=128):
        rk = []
        if key_bit_length == 128:
            T = [mk[0], mk[1], mk[2], mk[3]]
            for i in range(nr):
                cons = delta[i % 4]
                T[0] = self.rol(T[0] + self.rol(cons, i), 1)
                T[1] = self.rol(T[1] + self.rol(cons, i + 1), 3)
                T[2] = self.rol(T[2] + self.rol(cons, i + 2), 6)
                T[3] = self.rol(T[3] + self.rol(cons, i + 3), 11)
                rk.append((T[0].copy(), T[1].copy(), T[2].copy(),
                           T[1].copy(), T[3].copy(), T[1].copy()))
        return rk

    def encrypt(self, p, rks):
        c = p
        for rk in rks:
            c = self.enc_one_round(c, rk)
        return c

    def convert_to_binary(self, arr):
        X = np.zeros((len(arr) * self.WORD_SIZE, len(arr[0])), dtype=np.uint8)
        for i in range(len(arr) * self.WORD_SIZE):
            index = i // self.WORD_SIZE
            offset = self.WORD_SIZE - (i % self.WORD_SIZE) - 1
            X[i] = (arr[index] >> offset) & 1
        return X.transpose()

    def make_train_data(self, n, nr, diff, key_bit_length=128):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        mk = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, n)
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
        rk = self.expand_key(mk, nr, key_bit_length)
        cl0, cl1, cl2, cl3 = self.encrypt((pl0, pl1, pl2, pl3), rk)
        cr0, cr1, cr2, cr3 = self.encrypt((pr0, pr1, pr2, pr3), rk)
        # Calculating back
        cl0 = self.ror(cl0, self.a)
        cl1 = self.rol(cl1, self.b)
        cl2 = self.rol(cl2, self.c)
        cr0 = self.ror(cr0, self.a)
        cr1 = self.rol(cr1, self.b)
        cr2 = self.rol(cr2, self.c)
        # Ciphertext pair
        X = [cl0, cl1, cl2, cl3, cr0, cr1, cr2, cr3]
        X = self.convert_to_binary(X)
        return (X, Y)
