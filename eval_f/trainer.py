'''
Source code from https://github.com/agohr/deep_speck.git
'''
import numpy as np
from time import perf_counter

from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import (Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation)
from keras.regularizers import l2

from train_ciphers import get_train_cipher


RESNET_CONFIG = {
    'speck32':  {'num_groups': 4, 'word_size': 16},
    'speck48':  {'num_groups': 4, 'word_size': 24},
    'speck64':  {'num_groups': 4, 'word_size': 32},
    'speck96':  {'num_groups': 4, 'word_size': 48},
    'speck128': {'num_groups': 4, 'word_size': 64},
    'lea':      {'num_groups': 8, 'word_size': 32},
    'chaskey':  {'num_groups': 8, 'word_size': 32},
    'siphash':  {'num_groups': 2, 'word_size': 64},
}


def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)


def make_resnet(num_groups, word_size, num_filters=32, d1=64, d2=64, ks=3, depth=1, reg_param=1e-5):
    input_dim = num_groups * word_size
    inp = Input(shape=(input_dim,))
    rs = Reshape((num_groups, word_size))(inp)
    perm = Permute((2, 1))(rs)
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same',
                   kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0
    for _ in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    flat = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return model


def change_diff_to_tuple(algorithm, num):
    if algorithm == 'siphash':
        return num

    if algorithm.startswith('speck'):
        word_sizes = {
            'speck32': 16, 'speck48': 24, 'speck64': 32,
            'speck96': 48, 'speck128': 64,
        }
        ws = word_sizes[algorithm]
        mask = (1 << ws) - 1
        t = [0, 0]
        for i in range(2):
            t[1 - i] = (num >> (i * ws)) & mask
        return t

    if algorithm == 'lea':
        t = [0, 0, 0, 0]
        mask = 0xFFFFFFFF
        for i in range(4):
            x = (num >> (i * 32)) & mask
            t[3 - i] = (((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) | ((x & 0xFF000000) >> 24))
        return t

    if algorithm == 'chaskey':
        t = [0, 0, 0, 0]
        mask = 0xFFFFFFFF
        for i in range(4):
            t[3 - i] = (num >> (i * 32)) & mask
        return t

    raise ValueError(f"Unknown algorithm: {algorithm}")


def train_single_diff(algorithm, param, diff_hex, train_round, num_epochs=10, batch_size=5000, data_size=10**7):

    diff = change_diff_to_tuple(algorithm, diff_hex)
    cipher = get_train_cipher(algorithm, param)

    config = RESNET_CONFIG[algorithm]
    net = make_resnet(config['num_groups'], config['word_size'], num_filters=32, d1=64, d2=64, depth=1, reg_param=1e-5)
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])

    lr = LearningRateScheduler(cyclic_lr(min(num_epochs, 10), 0.002, 0.0001))

    if train_round == int(train_round):
        train_round = int(train_round)

    # Generate training and validation data
    X, Y = cipher.make_train_data(data_size, train_round, diff)
    X_eval, Y_eval = cipher.make_train_data(data_size // 10, train_round, diff)

    # Train
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size, validation_data=(X_eval, Y_eval), callbacks=[lr], verbose=0)
    del X, Y, X_eval, Y_eval

    raw_acc = np.max(h.history['val_acc'])
    normalized_acc = raw_acc * 2 - 1
    return normalized_acc


def train_for_diffs(algorithm, param, diff_hex_list, train_round, max_diffs=4, num_epochs=10, batch_size=5000, data_size=10**7):
    # Limit to max_diffs
    diffs_to_train = diff_hex_list[:max_diffs]

    results = []
    for diff_hex in diffs_to_train:
        start = perf_counter()
        acc = train_single_diff(algorithm, param, diff_hex, train_round, num_epochs=num_epochs, batch_size=batch_size, data_size=data_size)
        end = perf_counter()
        time_cost = round(end - start, 2)
        results.append((diff_hex, round(acc, 5), time_cost))
        print(f"  diff={hex(diff_hex)}, acc={round(acc, 5)}, time={time_cost}s")

    return results
