import numpy as np
from os import urandom
from ciphers.cipher import Cipher
from util import gen_diffs

def randomize_one_block(X, block_idx, block_size):
    new_X = X.copy()
    new_X[:, block_idx * block_size : (block_idx + 1) * block_size] = np.frombuffer(urandom(len(X) * block_size), dtype=np.uint8).reshape((len(X), block_size)) & 1
    return new_X

# Difference Sensitivity Test identifies the best difference
def detect_diff_importance(model, cipher: Cipher, diff_indexes: list, n: int, num_rounds):
    diffs = gen_diffs(diff_indexes, cipher.WORD_SIZE, cipher.WORD_NUM)
    X, Y = cipher.make_train_data_multidiff(n, num_rounds, diffs, 'only_diff')
    Z = (model.predict(X, batch_size=10000, verbose=0) > 0.5).flatten()
    ori_acc = np.sum(Z == Y) / n
    print("Origin model acc is", ori_acc)
    acc_drops = []
    for idx, diff_index in enumerate(diff_indexes):
        randomized_X = randomize_one_block(X, idx, cipher.WORD_NUM * cipher.WORD_SIZE)
        Z = (model.predict(randomized_X, batch_size=10000, verbose=0) > 0.5).flatten()
        acc = np.sum(Z == Y) / n
        print(f"Diff index is {diff_index}. Acc drop is {ori_acc - acc}")
        acc_drops.append(ori_acc - acc)
    return acc_drops