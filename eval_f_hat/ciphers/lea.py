import numpy as np
from os import urandom

delta = [0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec, 0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957]
delta = np.array(delta, dtype=np.uint32).reshape(8, 1)

class LEA(object):
    def __init__(self, a, b, c):
        self.WORD_SIZE = 32
        self.BLOCK_SIZE = 128
        self.MASK_VAL = (2**self.WORD_SIZE) - 1
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

    def dec_one_round(self, c, rk):
        p0 = c[3].copy()
        p1 = rk[1] ^ (self.ror(c[0], self.a) - (p0 ^ rk[0]))
        p2 = rk[3] ^ (self.ror(c[1], self.b) - (p1 ^ rk[2]))
        p3 = rk[5] ^ (self.ror(c[2], self.c) - (p2 ^ rk[4]))
        return (p0, p1, p2, p3)

    def expand_key(self, mk, nr, key_bit_length=128):
        assert key_bit_length in (128, 192, 256)
        rk = []
        if key_bit_length == 128:
            T = [mk[0], mk[1], mk[2], mk[3]]
            for i in range(nr):
                cons = delta[i % 4]
                T[0] = self.rol(T[0] + self.rol(cons, i), 1)
                T[1] = self.rol(T[1] + self.rol(cons, i + 1), 3)
                T[2] = self.rol(T[2] + self.rol(cons, i + 2), 6)
                T[3] = self.rol(T[3] + self.rol(cons, i + 3), 11)
                rk.append((T[0].copy(), T[1].copy(), T[2].copy(), T[1].copy(), T[3].copy(), T[1].copy()))
        elif key_bit_length == 192:
            T = [mk[0], mk[1], mk[2], mk[3], mk[4], mk[5]]
            for i in range(nr):
                cons = delta[i % 6]
                T[0] = self.rol(T[0] + self.rol(cons, i), 1)
                T[1] = self.rol(T[1] + self.rol(cons, i + 1), 3)
                T[2] = self.rol(T[2] + self.rol(cons, i + 2), 6)
                T[3] = self.rol(T[3] + self.rol(cons, i + 3), 11)
                T[4] = self.rol(T[4] + self.rol(cons, i + 4), 13)
                T[5] = self.rol(T[5] + self.rol(cons, i + 5), 17)
                rk.append((T[0].copy(), T[1].copy(), T[2].copy(), T[3].copy(), T[4].copy(), T[5].copy()))
        else:
            T = [mk[0], mk[1], mk[2], mk[3], mk[4], mk[5], mk[6], mk[7]]
            for i in range(nr):
                cons = delta[i % 8]
                T[(6 * i) % 8] = self.rol(T[(6 * i) % 8] + self.rol(cons, i), 1)
                T[(6 * i + 1) % 8] = self.rol(T[(6 * i + 1) % 8] + self.rol(cons, i + 1), 3)
                T[(6 * i + 2) % 8] = self.rol(T[(6 * i + 2) % 8] + self.rol(cons, i + 2), 6)
                T[(6 * i + 3) % 8] = self.rol(T[(6 * i + 3) % 8] + self.rol(cons, i + 3), 11)
                T[(6 * i + 4) % 8] = self.rol(T[(6 * i + 4) % 8] + self.rol(cons, i + 4), 13)
                T[(6 * i + 5) % 8] = self.rol(T[(6 * i + 5) % 8] + self.rol(cons, i + 5), 17)
                rk.append(((T[(6 * i) % 8].copy(), T[(6 * i + 1) % 8].copy(), T[(6 * i + 2) % 8].copy(), T[(6 * i + 3) % 8].copy(), T[(6 * i + 4) % 8].copy(), T[(6 * i + 5) % 8].copy())))
        return rk

    def encrypt(self, p, rks):
        c = p
        for rk in rks:
            c = self.enc_one_round(c, rk)
        return c

    def decrypt(self, c, rks):
        p = c
        for rk in reversed(rks):
            p = self.dec_one_round(p, rk)
        return p

    def convert_to_binary(self, arr):
        X = np.zeros((len(arr) * self.WORD_SIZE, len(arr[0])), dtype=np.uint8)
        for i in range(len(arr) * self.WORD_SIZE):
            index = i // self.WORD_SIZE
            offset = self.WORD_SIZE - (i % self.WORD_SIZE) - 1
            X[i] = (arr[index] >> offset) & 1
        X = X.transpose()
        return X

    def make_train_data(self, n, nr, diff, data_form='only_diff', key_bit_length=128):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        mk = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, n)
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
        rk = self.expand_key(mk, nr, key_bit_length)
        cl0, cl1, cl2, cl3 = self.encrypt((pl0, pl1, pl2, pl3), rk)
        cr0, cr1, cr2, cr3 = self.encrypt((pr0, pr1, pr2, pr3), rk)
        cl0 = self.ror(cl0, self.a)  # Calculating back
        cl1 = self.rol(cl1, self.b)
        cl2 = self.rol(cl2, self.c)
        cr0 = self.ror(cr0, self.a)
        cr1 = self.rol(cr1, self.b)
        cr2 = self.rol(cr2, self.c)
        if data_form == 'only_diff':
            d0, d1, d2, d3 = cl0 ^ cr0, cl1 ^ cr1, cl2 ^ cr2, cl3 ^ cr3
            X = [d0, d1, d2, d3]
        X = self.convert_to_binary(X)
        return (X, Y)

    def make_train_data_multidiff(self, n, nr, diffs, data_form='only_diff', key_bit_length=128):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        mk = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, n)
        p = [np.frombuffer(urandom(4 * n), dtype=np.uint32) for _ in range(4)]
        paired_p = []
        for diff in diffs:
            paired_p.append([p[i] ^ diff[i] for i in range(4)])
        negative_pos = (Y == 0)
        num_rand_samples = np.sum(negative_pos)
        for i in range(len(diffs)):
            for j in range(4):
                paired_p[i][j][negative_pos] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        rk = self.expand_key(mk, nr, key_bit_length)
        c = self.encrypt(p, rk)
        paired_c = [self.encrypt(x, rk) for x in paired_p]
        if data_form == 'only_diff':
            diff_words = []
            for x in paired_c:
                diff_words += [c[i] ^ x[i] for i in range(4)]
            X = diff_words
        X = self.convert_to_binary(X)
        return (X, Y)

    @classmethod
    def check_testvectors(cls, version=128):
        # Bit string is represented with Big-Endian mode
        cipher = cls(a=9, b=27, c=29)

        if version == 128:
            # test_vector for LEA-128
            key_bit_length = 128
            nr = 24
            key = (0x3c2d1e0f, 0x78695a4b, 0xb4a59687, 0xf0e1d2c3)
            plaintext = np.array((0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c), dtype=np.uint32)
            ciphertext = (0x354ec89f, 0x18c6c628, 0xa7c73255, 0xfd8b6404)

        if version == 192:
            # test_vector for LEA-192
            key_bit_length = 192
            nr = 28
            key = (0x53af3714, 0x75bd6930, 0x0c56c125, 0xa1d2ba78, 0x1c6734e5, 0x7cf27e00)
            plaintext = np.array((0xcbf4b41c, 0x51db4b6c, 0x0984ea68, 0x51fd7b72), dtype=np.uint32)
            ciphertext = (0x6d5c7269, 0xb7f812f9, 0xe611b50e, 0x70583c66)

        if version == 256:
            # test_vector for LEA-256
            key_bit_length = 256
            nr = 32
            key = (0xe279674f, 0x19931ebd, 0xac1530c6, 0xa7d7efff, 0x59edf091, 0x07701bdf, 0xe282fe69, 0x358c66f0)
            plaintext = np.array((0xe3ca31dc, 0x110a5eda, 0x20b066c9, 0xdefecfd7), dtype=np.uint32)
            ciphertext = (0x2004a2ed, 0xe867f698, 0xb82da057, 0xf2dfa7ca)

        ks = cipher.expand_key(key, nr=nr, key_bit_length=key_bit_length)
        c = cipher.encrypt(plaintext, ks)
        if c == ciphertext:
            if key_bit_length == 128:
                print('Test vector of LEA-128 is verified.')
            elif key_bit_length == 192:
                print('Test vector of LEA-192 is verified.')
            else:
                print('Test vector of LEA-256 is verified.')
        p = cipher.decrypt(c, ks)


if __name__ == '__main__':
    LEA.check_testvectors(version=128)
    LEA.check_testvectors(version=192)
    LEA.check_testvectors(version=256)