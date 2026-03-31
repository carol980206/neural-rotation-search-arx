import numpy as np
from keras.callbacks import LearningRateScheduler
from ciphers.cipher import Cipher
from eval_diff_score import get_good_diff, get_random_bias_for_reference
from detect_important_diff import detect_diff_importance
from util import gen_diffs, gen_diff, gen_diff_index
import model

def multidiff_net_train(cipher: Cipher, num_epochs: int, num_rounds, diffs: list, bs=4000, data_size=2*(10**6), data_form='only_diff', staged=False):
    if data_form == 'only_diff':
        num_words = cipher.WORD_NUM
    net = model.make_resnet(num_words=num_words * len(diffs), num_filters=32, d1=64, d2=64, word_size=cipher.WORD_SIZE, depth=1, reg_param=10**-5)
    net.compile(optimizer='adam',loss='mse',metrics=['acc'])
    lr = LearningRateScheduler(model.cyclic_lr(min(num_epochs, 10), 0.001, 0.0001))
    if staged:
        X, Y = cipher.make_train_data_multidiff(data_size, num_rounds - 1, diffs, data_form)
        X_eval, Y_eval = cipher.make_train_data_multidiff(data_size // 10, num_rounds - 1, diffs, data_form)
        h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr], verbose=0)
        del X; del Y; del X_eval; del Y_eval
        X, Y = cipher.make_train_data_multidiff(data_size, num_rounds, diffs, data_form)
        X_eval, Y_eval = cipher.make_train_data_multidiff(data_size // 10, num_rounds, diffs, data_form)
        h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr], verbose=0)
        del X; del Y; del X_eval; del Y_eval
    else:
        X, Y = cipher.make_train_data_multidiff(data_size, num_rounds, diffs, data_form)
        X_eval, Y_eval = cipher.make_train_data_multidiff(data_size // 10, num_rounds, diffs, data_form)
        h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr], verbose=0)
        del X; del Y; del X_eval; del Y_eval
    return net

def select_best_diff_index(cipher: Cipher, diff_indexes, num_rounds):
    diffs = gen_diffs(diff_indexes, cipher.WORD_SIZE, cipher.WORD_NUM)
    model = multidiff_net_train(cipher, 5, num_rounds, diffs, 5000, 10**7, 'only_diff', False)
    diff_importance = detect_diff_importance(model, cipher, diff_indexes, 10**6, num_rounds)
    return diff_indexes[np.argmax(diff_importance)]

def get_top_input_diffs(cipher: Cipher, T: int, num_rounds):
    if cipher.cipher_name == 'speck128':
        diff_indexes = [[i] for i in range(128)] + [[63, 64 + ((cipher.tweakable_params[0] - 1) % cipher.WORD_SIZE)]]
    elif cipher.cipher_name == 'speck96':
        diff_indexes = [[i] for i in range(96)] + [[47, 48 + ((cipher.tweakable_params[0] - 1) % cipher.WORD_SIZE)]]
    elif cipher.cipher_name == 'speck64':
        diff_indexes = [[i] for i in range(64)] + [[31, 32 + ((cipher.tweakable_params[0] - 1) % cipher.WORD_SIZE)]]
    elif cipher.cipher_name == 'speck48':
        diff_indexes = [[i] for i in range(48)] + [[23, 24 + ((cipher.tweakable_params[0] - 1) % cipher.WORD_SIZE)]]
    elif cipher.cipher_name == 'speck32':
        diff_indexes = [[i] for i in range(32)] + [[15, 16 + ((cipher.tweakable_params[0] - 1) % cipher.WORD_SIZE)]]
    elif cipher.cipher_name == 'lea':
        diff_indexes = [[i] for i in range(128)] + [[31,63], [31,95], [31,127], [63,95], [63,127], [95,127], [31,63,95], [31,63,127], [31,95,127], [63,95,127], [31,63,95,127]]
    elif cipher.cipher_name == 'chaskey':
        diff_indexes = [[i] for i in range(128)] + [[31,63], [31,95], [31,127], [63,95], [63,127], [95,127], [31,63,95], [31,63,127], [31,95,127], [63,95,127], [31,63,95,127]]
    elif cipher.cipher_name == 'siphash':
        diff_indexes = [[i] for i in range(64)]
        
    diffs = gen_diffs(diff_indexes, cipher.WORD_SIZE, cipher.WORD_NUM)
    n = 1 << 20
    random_score = get_random_bias_for_reference(n)
    good_diff = get_good_diff(cipher, diffs, n, int(num_rounds-1))[:T]  # Obtain the differences of the first T high bias
    good_diff_indexes = [gen_diff_index(diff[0], cipher.WORD_SIZE, cipher.WORD_NUM) for diff in good_diff]
    print("Random score is", random_score)
    for i, diff_index in enumerate(good_diff_indexes):
        print(f"Diff index is {diff_index}. Score is {good_diff[i][1]}.")
    return gen_diff(select_best_diff_index(cipher, good_diff_indexes, num_rounds), cipher.WORD_SIZE, cipher.WORD_NUM)

def train_with_good_diff(cipher: Cipher, num_epochs: int, num_rounds, T: int, bs=4000, data_size=2*(10**6), data_form='only_diff', staged=False):
    if data_form == 'only_diff':
        num_words = cipher.WORD_NUM
    diff = get_top_input_diffs(cipher, T, num_rounds)
    print("Find best diff:", gen_diff_index(diff, cipher.WORD_SIZE, cipher.WORD_NUM))
    net = model.make_resnet(num_words=num_words, num_filters=32, d1=64, d2=64, word_size=cipher.WORD_SIZE, depth=1, reg_param=10**-5)
    net.compile(optimizer='adam',loss='mse',metrics=['acc'])
    lr = LearningRateScheduler(model.cyclic_lr(min(num_epochs, 10), 0.002, 0.0001))
    if staged:
        X, Y = cipher.make_train_data(data_size, num_rounds - 1, diff, data_form)
        X_eval, Y_eval = cipher.make_train_data(data_size // 10, num_rounds - 1, diff, data_form)
        h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr], verbose=0)
        del X, Y, X_eval, Y_eval
        X, Y = cipher.make_train_data(data_size, num_rounds, diff, data_form)
        X_eval, Y_eval = cipher.make_train_data(data_size // 10, num_rounds, diff, data_form)
        h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr], verbose=0)
        del X, Y, X_eval, Y_eval
    else:
        X, Y = cipher.make_train_data(data_size, num_rounds, diff, data_form)
        X_eval, Y_eval = cipher.make_train_data(data_size // 10, num_rounds, diff, data_form)
        h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr], verbose=0)
        del X, Y, X_eval, Y_eval
    acc = np.max(h.history['val_acc'])
    print("This try acc is", acc)
    return acc