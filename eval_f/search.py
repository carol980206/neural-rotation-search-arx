'''
Source code from https://github.com/Crypto-TII/AutoND.git
'''
import os
import importlib
import optimizer


def findGoodInputDifferences(cipher_name, scenario, param, max_round, output_dir, epsilon=0.1, dual_key=False, T=0.008, round_step=1):
    cipher = importlib.import_module('ciphers.' + cipher_name, package='ciphers')
    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    encryption_function = cipher.encrypt
    best_differences, highest_round = optimizer.optimize(
        plain_bits, key_bits, encryption_function, param, max_round,
        scenario=scenario, log_file=None, epsilon=epsilon,
        dual_key=dual_key, T=T, round_step=round_step)
    return best_differences, highest_round
