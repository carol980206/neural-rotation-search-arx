import numpy as np
import os
from ciphers.cipher import Cipher
from keras.callbacks import LearningRateScheduler
from keras.models import Model, load_model
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import Adam

def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)

# Make residual tower of convolutional blocks
def make_resnet(num_words=4, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=32, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(num_words * word_size,))
    rs = Reshape((num_words, word_size))(inp)
    perm = Permute((2,1))(rs)
    # Add a single residual layer that will expand the data to num_filters channels
    # This is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    # Add residual blocks
    shortcut = conv0
    for _ in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2
    # Add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return model