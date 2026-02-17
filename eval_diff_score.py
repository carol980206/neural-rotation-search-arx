from os import urandom
import numpy as np
from ciphers.cipher import Cipher

def get_diff_score(cipher: Cipher, diff, p, rks, c):
    pr = [p[i] ^ diff[i] for i in range(cipher.WORD_NUM)]
    cr = cipher.encrypt(pr, rks)
    X = cipher.convert_to_binary([c[i] ^ cr[i] for i in range(cipher.WORD_NUM)])
    return np.average(np.abs(np.sum(X, axis=0) / len(X) - 0.5))
    
def get_good_diff(cipher: Cipher, test_diffs, n, num_rounds):
    diff_and_score = []
    p = cipher.make_plaintexts(n)
    rks = cipher.make_rks(n, num_rounds)
    c = cipher.encrypt(p, rks)
    for diff in test_diffs:
        score = get_diff_score(cipher, diff, p, rks, c)
        diff_and_score.append((diff, score))
    diff_and_score = sorted(diff_and_score, key=lambda x: x[1], reverse=True)
    return diff_and_score

def get_random_bias_for_reference(n):
    X = np.frombuffer(urandom(n * 128), dtype=np.uint8).reshape(128, n) & 0x1
    bit_bias = np.sum(X, axis=1) / n - 0.5
    score = np.sum(np.abs(bit_bias))
    return score