import numpy as np
from os import urandom

class Speck32(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.WORD_SIZE = 16
        self.BLOCK_SIZE = 32
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

    def expand_key(self, k, t, version=64):
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
    def check_testvector(cls, version=64):
        cipher = cls(alpha=7, beta=2)

        if version == 64:
            key = (0x1918,0x1110,0x0908,0x0100)
            pt = (0x6574, 0x694c)
            ks = cipher.expand_key(key, 22, version=version)
            ct = cipher.encrypt(pt, ks)
            if ct == (0xa868, 0x42f2):
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
    
    def make_train_data(self, n, nr, diff, data_form='only_diff', key_bit_length=64):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        mk_num = key_bit_length // self.WORD_SIZE
        mk = np.frombuffer(urandom(n * 2 * mk_num), dtype=np.uint16).reshape((mk_num, n))
        rk = self.expand_key(mk, nr, key_bit_length)
        p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p1l = p0l ^ diff[0]; p1r = p0r ^ diff[1]
        rand_sample_pos = (Y == 0)
        num_rand_samples = np.sum(rand_sample_pos)
        p1l[rand_sample_pos] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        p1r[rand_sample_pos] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        c0l, c0r = self.encrypt((p0l, p0r), rk)
        c1l, c1r = self.encrypt((p1l, p1r), rk)
        c0r = self.ror((c0l ^ c0r), self.beta)  # Calculating back
        c1r = self.ror((c1l ^ c1r), self.beta)
        if data_form == 'only_diff':
            dl, dr = c0l ^ c1l, c0r ^ c1r
            X = [dl, dr]
        X = self.convert_to_binary(X)
        return (X, Y)
    
    def make_train_data_multidiff(self, n, nr, diffs, data_form='only_diff', version=64):
        mk_num = version // self.WORD_SIZE
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        mk = np.frombuffer(urandom(n * 2 * mk_num), dtype=np.uint16).reshape((mk_num, n))
        p = [np.frombuffer(urandom(2 * n), dtype=np.uint16) for _ in range(2)]
        paired_p = []
        for diff in diffs:
            paired_p.append([p[0] ^ diff[0], p[1] ^ diff[1]])
        negative_pos = (Y == 0)
        num_rand_samples = np.sum(negative_pos)
        for i in range(len(diffs)):
            paired_p[i][0][negative_pos] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
            paired_p[i][1][negative_pos] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
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
    Speck32.check_testvector(version=64)