from ciphers.lea import LEA
from ciphers.speck128 import Speck128
from ciphers.speck96 import Speck96
from ciphers.speck64 import Speck64
from ciphers.speck48 import Speck48
from ciphers.speck32 import Speck32
from ciphers.chaskey import Chaskey
from ciphers.siphash import Siphash
from copy import copy
from os import urandom
import numpy as np
import gc

SUPPORTED_CIPHERS = ['lea', 'speck128', 'speck96', 'speck64', 'speck48', 'speck32', 'chaskey', 'siphash']

class Cipher(object):
    """
    A cipher wrapper that provides unified interface.
    """
    def __init__(self, cipher_name: str, tweakable_params: list, master_key_bit_length=None):
        assert cipher_name in SUPPORTED_CIPHERS
        self.master_key_bit_length = master_key_bit_length
        self.cipher_name = cipher_name
        self.tweakable_params = copy(tweakable_params)
        if cipher_name == 'lea':
            assert len(tweakable_params) == 3
            self.cipher_engine = LEA(tweakable_params[0], tweakable_params[1], tweakable_params[2])
            self.WORD_SIZE = 32
            self.WORD_NUM = 4
            if self.master_key_bit_length is None:
                self.master_key_bit_length = 128
        elif cipher_name == 'speck128':
            assert len(tweakable_params) == 2
            self.cipher_engine = Speck128(tweakable_params[0], tweakable_params[1])
            self.WORD_SIZE = 64
            self.WORD_NUM = 2
            if self.master_key_bit_length is None:
                self.master_key_bit_length = 128
        elif cipher_name == 'speck96':
            assert len(tweakable_params) == 2
            self.cipher_engine = Speck96(tweakable_params[0], tweakable_params[1])
            self.WORD_SIZE = 48
            self.WORD_NUM = 2
            if self.master_key_bit_length is None:
                self.master_key_bit_length = 96
        elif cipher_name == 'speck64':
            assert len(tweakable_params) == 2
            self.cipher_engine = Speck64(tweakable_params[0], tweakable_params[1])
            self.WORD_SIZE = 32
            self.WORD_NUM = 2
            if self.master_key_bit_length is None:
                self.master_key_bit_length = 96
        elif cipher_name == 'speck48':
            assert len(tweakable_params) == 2
            self.cipher_engine = Speck48(tweakable_params[0], tweakable_params[1])
            self.WORD_SIZE = 24
            self.WORD_NUM = 2
            if self.master_key_bit_length is None:
                self.master_key_bit_length = 72
        elif cipher_name == 'speck32':
            assert len(tweakable_params) == 2
            self.cipher_engine = Speck32(tweakable_params[0], tweakable_params[1])
            self.WORD_SIZE = 16
            self.WORD_NUM = 2
            if self.master_key_bit_length is None:
                self.master_key_bit_length = 64
        elif cipher_name == 'chaskey':
            assert len(tweakable_params) == 6
            self.cipher_engine = Chaskey(tweakable_params)
            self.WORD_SIZE = 32
            self.WORD_NUM = 4
            self.master_key_bit_length = 128
        elif cipher_name == 'siphash':
            assert len(tweakable_params) == 6
            self.cipher_engine = Siphash(tweakable_params)
            self.WORD_SIZE = 64
            self.WORD_NUM = 1
            self.master_key_bit_length = 128

    def make_plaintexts(self, n):
        if self.WORD_SIZE == 16:
            word_dtype = np.uint16
        elif self.WORD_SIZE == 32:
            word_dtype = np.uint32
        elif self.WORD_SIZE == 64:
            word_dtype = np.uint64
        elif self.WORD_SIZE == 24:
            p = np.frombuffer(urandom(n * 2 * 4), dtype=np.uint32)
            p = p & (1 << self.WORD_SIZE) - 1
            p = np.reshape(p, (self.WORD_NUM, n))
            return p 
        elif self.WORD_SIZE == 48:
            p = np.frombuffer(urandom(n * 2 * 8), dtype=np.uint64)
            p = p & (1 << self.WORD_SIZE) - 1
            p = np.reshape(p, (self.WORD_NUM, n))
            return p
        bytes_per_word = self.WORD_SIZE // 8
        p = np.frombuffer(urandom(n * self.WORD_NUM * bytes_per_word), dtype=word_dtype)
        p = np.reshape(p, (self.WORD_NUM, n))
        return p
    
    def make_rks(self, n, nr):
        if self.WORD_SIZE == 16:
            word_dtype = np.uint16
        elif self.WORD_SIZE == 32:
            word_dtype = np.uint32
        elif self.WORD_SIZE == 64:
            word_dtype = np.uint64
        elif self.WORD_SIZE == 24:
            mk_word_num = self.master_key_bit_length // self.WORD_SIZE
            mk = np.frombuffer(urandom(n * mk_word_num * 4), dtype=np.uint32)
            mk = mk & (1 << self.WORD_SIZE) - 1
            mk = np.reshape(mk, (mk_word_num, n))
            return self.cipher_engine.expand_key(mk, nr, self.master_key_bit_length)
        elif self.WORD_SIZE == 48:
            mk_word_num = self.master_key_bit_length // self.WORD_SIZE
            mk = np.frombuffer(urandom(n * mk_word_num * 8), dtype=np.uint64)
            mk = mk & (1 << self.WORD_SIZE) - 1
            mk = np.reshape(mk, (mk_word_num, n))
            return self.cipher_engine.expand_key(mk, nr, self.master_key_bit_length)
        bytes_per_word = self.WORD_SIZE // 8
        mk_word_num = self.master_key_bit_length // self.WORD_SIZE
        mk = np.frombuffer(urandom(n * mk_word_num * bytes_per_word), dtype=word_dtype)
        mk = np.reshape(mk, (mk_word_num, n))
        return self.cipher_engine.expand_key(mk, nr, self.master_key_bit_length)
    
    def make_train_data(self, n, nr, diff, data_form):
        return self.cipher_engine.make_train_data(n, nr, diff, data_form, self.master_key_bit_length)
    
    def convert_to_binary(self, X):
        return self.cipher_engine.convert_to_binary(X)
    
    def encrypt(self, p, rks):
        return self.cipher_engine.encrypt(p, rks)
    
    def make_train_data_multidiff(self, n, nr, diffs, data_form):
        Xs = []
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 0x1
        for diff in diffs:
            gc.collect()
            p = self.make_plaintexts(n)
            rks = self.make_rks(n, nr)
            Xs.append(self.make_train_data_from_given_sample(p, rks, Y, diff, data_form))
            del p; del rks
        Xs = np.concatenate(Xs, axis=1)
        return Xs, Y

    def make_train_data_from_given_sample(self, p, rk, Y, diff, data_form):
        if self.WORD_SIZE == 16:
            word_dtype = np.uint16
        elif self.WORD_SIZE == 32 or self.WORD_SIZE == 24:
            word_dtype = np.uint32
        elif self.WORD_SIZE == 64 or self.WORD_SIZE == 48:
            word_dtype = np.uint64
        bytes_per_word = self.WORD_SIZE // 8
        if self.WORD_SIZE == 24:
            bytes_per_word = 4
        if self.WORD_SIZE == 48:
            bytes_per_word = 8
        pr = [p[i] ^ diff[i] for i in range(self.WORD_NUM)]
        rand_pos = (Y == 0)
        num_rand_samples = np.sum(rand_pos)
        for i in range(self.WORD_NUM):
            pr[i][rand_pos] = np.frombuffer(urandom(bytes_per_word * num_rand_samples), dtype=word_dtype)
        c = self.cipher_engine.encrypt(p, rk)
        cr = self.cipher_engine.encrypt(pr, rk)
        if data_form == 'only_diff':
            X = [c[i] ^ cr[i] for i in range(self.WORD_NUM)]
        X = self.cipher_engine.convert_to_binary(X)
        return X
    
    def make_target_diff_data(self, n, nr, diff, data_form='only_diff', binary_form=False):
        rks = self.make_rks(n, nr)
        p0 = self.make_plaintexts(n)
        p1 = [p0[i] ^ diff[i] for i in range(self.WORD_NUM)]
        c0 = self.cipher_engine.encrypt(p0, rks)
        c1 = self.cipher_engine.encrypt(p1, rks)
        if data_form == 'only_diff':
            X = [c0[i] ^ c1[i] for i in range(self.WORD_NUM)]
        if binary_form:
            X = self.cipher_engine.convert_to_binary(X)
        return X, rks