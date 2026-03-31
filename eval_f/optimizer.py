'''
Source code from https://github.com/Crypto-TII/AutoND.git
'''
import os
from os import urandom
import numpy as np
import pandas as pd

NUM_GENERATIONS = 50
NUM_SAMPLES = 10**4


def bitArrayToIntegers(arr):
    packed = np.packbits(arr, axis=1)
    return [int.from_bytes(x.tobytes(), 'big') for x in packed]


def evaluate_multiple_differences(candidate_differences, pt0, keys0, C0, nr, param, plain_bits, key_bits, encrypt, scenario="single-key", dual_key=False, keys1=None):
    dp = candidate_differences[:, :plain_bits]
    pt1 = (np.broadcast_to(dp[:, None, :], (len(candidate_differences), len(pt0), plain_bits)) ^ pt0).reshape(-1, plain_bits)
    if scenario == "related-key":
        dk = candidate_differences[:, plain_bits:]
    else:
        dk = np.zeros((len(candidate_differences), key_bits), dtype=np.uint8)
    keys0_diff = (np.broadcast_to(dk[:, None, :], (len(candidate_differences), len(keys0), key_bits)) ^ keys0).reshape(-1, key_bits)
    if dual_key:
        keys1_diff = (np.broadcast_to(dk[:, None, :], (len(candidate_differences), len(keys1), key_bits)) ^ keys1).reshape(-1, key_bits)
        C1 = encrypt(pt1, keys0_diff, keys1_diff, nr, param)
    else:
        C1 = encrypt(pt1, keys0_diff, nr, param)
    differences_in_output = C1.reshape(len(candidate_differences), len(pt0), -1) ^ C0
    scores = np.average(np.abs(0.5 - np.average(differences_in_output, axis=1)), axis=1)
    zero_diffs = np.where(np.sum(candidate_differences, axis=1) == 0)
    scores[zero_diffs] = 0
    return scores


def evo(f, n=NUM_GENERATIONS, num_bits=32, L=32, gen=None, verbose=0):
    mutProb = 100
    if gen is None:
        gen = np.random.randint(2, size=(L**2, num_bits), dtype=np.uint8)
    scores = f(gen)
    idx = np.arange(len(gen))
    explored = np.copy(gen)
    good = idx[np.argsort(scores)][-L:]
    gen = gen[good]
    scores = scores[good]
    cpt = len(gen)
    for generation in range(n):
        kids = np.array([gen[i] ^ gen[j] for i in range(len(gen)) for j in range(i+1, len(gen))], dtype=np.uint8)
        selected = np.where(np.random.randint(0, 100, len(kids)) > (100 - mutProb))
        numMut = len(selected[0])
        tmp = kids[selected].copy()
        kids[selected[0].tolist(), np.random.randint(num_bits, size=numMut)] ^= 1
        kids = np.unique(kids[(kids[:, None] != explored).any(-1).all(-1)], axis=0)
        explored = np.vstack([explored, kids])
        cpt += len(kids)
        if len(kids) > 0:
            scores = np.append(scores, f(kids))
            gen = np.vstack([gen, kids])
            idx = np.arange(len(gen))
            good = idx[np.argsort(scores)][-L:]
            gen = gen[good]
            scores = scores[good]
        if verbose:
            genInt = np.packbits(gen[-4:, :], axis=1)
            hexGen = [hex(int.from_bytes(x.tobytes(), 'big')) for x in genInt]
            print(f'Generation {generation}/{n}, {cpt} nodes explored, {len(gen)} current, best is {[x for x in hexGen]} with {scores[-4:]}', flush=True)
        if np.all(scores == 0.5):
            break
    return gen, scores


def DataframeFromSortedDifferences(differences, scores, scenario, plain_bits, key_bits=0):
    idx = np.arange(len(differences))
    good = idx[np.argsort(scores)]
    sorted_diffs = differences[good]
    sorted_scores = scores[good].round(4)
    diffs_to_print = bitArrayToIntegers(sorted_diffs)
    data = []
    for idx, d in enumerate(diffs_to_print):
        if scenario == "related-key":
            data.append([({hex(d >> key_bits)}, {hex(d & (2**key_bits - 1))}), {sorted_scores[idx]}])
        else:
            data.append([{hex(d)}, {sorted_scores[idx]}])
    df = pd.DataFrame(data, columns=['Difference', 'Weighted score'])
    return df


def PrettyPrintBestEpsilonCloseDifferences(differences, scores, epsilon, scenario, plain_bits, key_bits=0):
    idx = np.arange(len(differences))
    order = idx[np.argsort(scores)]
    sorted_diffs = differences[order]
    sorted_scores = scores[order].round(4)
    best_score = sorted_scores[-1]
    threshold = best_score * (1 - epsilon)
    keep = np.where(sorted_scores > threshold)
    diffs_to_print = bitArrayToIntegers(sorted_diffs[keep])
    scores_to_print = sorted_scores[keep]
    resStr = ''
    for idx, d in enumerate(diffs_to_print):
        if scenario == "related-key":
            resStr = resStr + f'[{hex(d)} ({hex(d >> key_bits)}, {hex(d & (2**key_bits - 1))}), {scores_to_print[idx]}]\n'
        else:
            resStr = resStr + f'[{hex(d)}, {scores_to_print[idx]}]\n'
    return resStr, sorted_diffs[keep], diffs_to_print


def PrettyPrintBestNDifferences(differences, scores, n, scenario, plain_bits, key_bits=0):
    idx = np.arange(len(differences))
    good = idx[np.argsort(scores)]
    sorted_diffs = differences[good]
    sorted_scores = scores[good].round(4)[-n:]
    diffs_to_print = bitArrayToIntegers(sorted_diffs)[-n:]
    resStr = ''
    for idx, d in enumerate(diffs_to_print):
        if scenario == "related-key":
            resStr = resStr + f'[{hex(d)} ({hex(d >> key_bits)}, {hex(d & (2**key_bits - 1))}), {sorted_scores[idx]}]\n'
        else:
            resStr = resStr + f'[{hex(d)}, {sorted_scores[idx]}]\n'
    return resStr, sorted_diffs[-n:], diffs_to_print


def optimize(plain_bits, key_bits, encryption_function, param, max_round, nb_samples=NUM_SAMPLES, scenario="single-key", log_file=None, epsilon=0.1, dual_key=False, T=0.008, round_step=1):
    allDiffs = None
    diffs = None
    current_round = round_step
    if scenario == "single-key":
        bits_to_search = plain_bits
    else:
        bits_to_search = plain_bits + key_bits
    while True:
        if round(current_round, 4) >= round(max_round, 4):
            highest_non_random_round = round(current_round - round_step, 4)
            break
        print("Evaluating differences at round ", current_round)
        keys0 = (np.frombuffer(urandom(nb_samples * key_bits), dtype=np.uint8) & 1).reshape(nb_samples, key_bits)
        pt0 = (np.frombuffer(urandom(nb_samples * plain_bits), dtype=np.uint8) & 1).reshape(nb_samples, plain_bits)
        keys1 = None
        if dual_key:
            keys1 = (np.frombuffer(urandom(nb_samples * key_bits), dtype=np.uint8) & 1).reshape(nb_samples, key_bits)
            C0 = encryption_function(pt0, keys0, keys1, current_round, param)
        else:
            C0 = encryption_function(pt0, keys0, current_round, param)
        diffs, scores = evo(
            f=lambda x: evaluate_multiple_differences(
                x, pt0, keys0, C0, current_round, param, plain_bits, key_bits,
                encryption_function, scenario=scenario, dual_key=dual_key, keys1=keys1),
            num_bits=bits_to_search, L=32, gen=diffs, verbose=0)
        if allDiffs is None:
            allDiffs = diffs
        else:
            allDiffs = np.concatenate([allDiffs, diffs])
        if scores[-1] < T:
            highest_non_random_round = round(current_round - round_step, 4)
            break
        current_round = round(current_round + round_step, 4)

    finalScores = {}
    allDiffs = np.unique(allDiffs, axis=0)
    cumulativeScores = np.zeros(len(allDiffs))
    weightedScores = np.zeros(len(allDiffs))
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f' \n New log start, reached round {str(highest_non_random_round)} \n')
            f.write(f'Params: {param}')
    nr = round_step
    while round(nr, 4) < round(current_round, 4):
        keys0 = (np.frombuffer(urandom(nb_samples * key_bits), dtype=np.uint8) & 1).reshape(nb_samples, key_bits)
        pt0 = (np.frombuffer(urandom(nb_samples * plain_bits), dtype=np.uint8) & 1).reshape(nb_samples, plain_bits)
        keys1 = None
        if dual_key:
            keys1 = (np.frombuffer(urandom(nb_samples * key_bits), dtype=np.uint8) & 1).reshape(nb_samples, key_bits)
            C0 = encryption_function(pt0, keys0, keys1, nr, param)
        else:
            C0 = encryption_function(pt0, keys0, nr, param)
        finalScores[nr] = evaluate_multiple_differences(
            allDiffs, pt0, keys0, C0, nr, param, plain_bits, key_bits,
            encryption_function, scenario=scenario, dual_key=dual_key, keys1=keys1)
        cumulativeScores += np.array(finalScores[nr])
        weightedScores += nr * np.array(finalScores[nr])

        result, _, _ = PrettyPrintBestNDifferences(allDiffs, finalScores[nr], 5, scenario, plain_bits, key_bits)
        resStr = f'Best at {nr}: \n{result}'
        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write(resStr)
        nr = round(nr + round_step, 4)

    result, _, _ = PrettyPrintBestNDifferences(allDiffs, cumulativeScores, 5, scenario, plain_bits, key_bits)
    resStr = f'Best Cumulative: \n{result}'
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(resStr)

    result, _, _ = PrettyPrintBestNDifferences(allDiffs, weightedScores, 5, scenario, plain_bits, key_bits)
    resStr = f'Best Weighted: \n{result}'
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(resStr)

    result, diffs_as_binary, diffs_as_hex = PrettyPrintBestEpsilonCloseDifferences(
        allDiffs, weightedScores, epsilon, scenario, plain_bits, key_bits)
    if log_file is not None:
        df = DataframeFromSortedDifferences(allDiffs, weightedScores, scenario, plain_bits, key_bits)
        df.to_csv(f'{log_file}_best_weighted_differences.csv')
    return (diffs_as_hex, highest_non_random_round)
