import numpy as np
from os import urandom

class Speck128(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.WORD_SIZE = 64
        self.BLOCK_SIZE = 128
        self.MASK_VAL = (2**self.WORD_SIZE) - 1
    
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
        return(c0,c1)

    def expand_key(self, k, t, version=128):
        ks = [0 for i in range(t)]
        ks[0] = k[len(k)-1]
        l = list(reversed(k[:len(k)-1]))

        ver = version // self.WORD_SIZE
        for i in range(t-1):
            l[i % (ver-1)], ks[i + 1] = self.enc_one_round((l[i % (ver-1)], ks[i]), i)
        return(ks)

    def encrypt(self, p, ks):
        x, y = p[0], p[1]
        for k in ks:
            x,y = self.enc_one_round((x,y), k)
        return(x, y)

    @classmethod
    def check_testvector(cls, version=128):
        cipher = cls(alpha=8, beta=3)

        if version == 128:
            key = (0x0f0e0d0c0b0a0908, 0x0706050403020100)
            pt = (0x6c61766975716520, 0x7469206564616d20)
            ks = cipher.expand_key(key, 32, version=version)
            ct = cipher.encrypt(pt, ks)
            if ct == (0xa65d985179783265, 0x7860fedf5c570d18):
                flag = 1
            else:
                flag = 0
        elif version == 192:
            key = (0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100)
            pt = (0x7261482066656968, 0x43206f7420746e65)
            ks = cipher.expand_key(key, 33, version=version)
            ct = cipher.encrypt(pt, ks)
            if ct == (0x1be4cf3a13135566, 0xf9bc185de03c1886):
                flag = 1
            else:
                flag = 0
        elif version == 256:
            key = (0x1f1e1d1c1b1a1918, 0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100)
            pt = (0x65736f6874206e49, 0x202e72656e6f6f70)
            ks = cipher.expand_key(key, 34, version=version)
            ct = cipher.encrypt(pt, ks)
            if ct == (0x4109010405c0f53e, 0x4eeeb48d9c188f43):
                flag = 1
            else:
                flag = 0
        if flag == 1:
            print("Testvector verified.")
            return (True)
        else:
            print("Testvector not verified.")
            return (False)

    def convert_to_binary(self, arr):
        n = len(arr[0])
        X = np.zeros((len(arr) * self.WORD_SIZE, n), dtype=np.uint8)
        for i in range(len(arr) * self.WORD_SIZE):
            index = i // self.WORD_SIZE
            offset = self.WORD_SIZE - (i % self.WORD_SIZE) - 1
            X[i] = (arr[index] >> offset) & 1
        X = X.transpose()
        return(X)
    
    def make_train_data(self, n, nr, diff, data_form='only_diff', key_bit_length=128):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        mk_num = key_bit_length // self.WORD_SIZE
        mk = np.frombuffer(urandom(n * 8 * mk_num), dtype=np.uint64).reshape((mk_num, n))
        rk = self.expand_key(mk, nr, key_bit_length)
        p0l = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        p0r = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        p1l = p0l ^ diff[0]; p1r = p0r ^ diff[1]
        rand_sample_pos = (Y == 0)
        num_rand_samples = np.sum(rand_sample_pos)
        p1l[rand_sample_pos] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
        p1r[rand_sample_pos] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
        c0l, c0r = self.encrypt((p0l, p0r), rk)
        c1l, c1r = self.encrypt((p1l, p1r), rk)
        c0r = self.ror((c0l ^ c0r), self.beta)  # Calculating back
        c1r = self.ror((c1l ^ c1r), self.beta)
        if data_form == 'only_diff':
            dl, dr = c0l ^ c1l, c0r ^ c1r
            X = [dl, dr]
        X = self.convert_to_binary(X)
        return (X, Y)
    
    def make_train_data_multidiff(self, n, nr, diffs, data_form='only_diff', version=128):
        mk_num = version // self.WORD_SIZE
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        mk = np.frombuffer(urandom(n * 8 * mk_num), dtype=np.uint64).reshape((mk_num, n))
        p = [np.frombuffer(urandom(8 * n), dtype=np.uint64) for _ in range(2)]
        paired_p = []
        for diff in diffs:
            paired_p.append([p[0] ^ diff[0], p[1] ^ diff[1]])
        negative_pos = (Y == 0)
        num_rand_samples = np.sum(negative_pos)
        for i in range(len(diffs)):
            paired_p[i][0][negative_pos] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
            paired_p[i][1][negative_pos] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
        rk = self.expand_key(mk, nr, version)
        c = self.encrypt(p, rk)
        paired_c = [self.encrypt(x, rk) for x in paired_p]
        if data_form == 'only_diff':
            diff_words = []
            for x in paired_c:
                diff_words += [c[i] ^ x[i] for i in range(2)]
        X = self.convert_to_binary(diff_words)
        return (X, Y)

if __name__ == "__main__":
    Speck128.check_testvector(version=128)
    Speck128.check_testvector(version=192)
    Speck128.check_testvector(version=256)