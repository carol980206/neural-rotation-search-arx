import os
import sys
import time
import json
import argparse
from multiprocessing import Pool
import search


# Algorithm name -> (cipher module name, dual_key, default bias threshold, round_step)
ALGORITHM_CONFIG = {
    'speck32':   {'cipher': 'speck3264',   'dual_key': False, 'T': 0.008, 'round_step': 1},
    'speck48':   {'cipher': 'speck4896',   'dual_key': False, 'T': 0.008, 'round_step': 1},
    'speck64':   {'cipher': 'speck64128',  'dual_key': False, 'T': 0.008, 'round_step': 1},
    'speck96':   {'cipher': 'speck96144',  'dual_key': False, 'T': 0.008, 'round_step': 1},
    'speck128':  {'cipher': 'speck128256', 'dual_key': False, 'T': 0.008, 'round_step': 1},
    'lea':       {'cipher': 'lea',         'dual_key': False, 'T': 0.008, 'round_step': 1},
    'chaskey':   {'cipher': 'chaskey',     'dual_key': False, 'T': 0.008, 'round_step': 0.5},
    'siphash':   {'cipher': 'siphash',     'dual_key': True,  'T': 0.008, 'round_step': 1},
}


def get_search_max_round(algorithm, max_round):
    if algorithm == 'speck32' and max_round == 7:
        return 6
    return max_round


def run_single_param(args_tuple):
    (algorithm, param, max_round, scenario, output_dir, epsilon, train_config) = args_tuple
    config = ALGORITHM_CONFIG[algorithm]
    cipher_name = config['cipher']
    dual_key = config['dual_key']
    T = config['T']
    round_step = config['round_step']

    # Compute search round (may differ from training round)
    search_max_round = get_search_max_round(algorithm, max_round)

    start = time.perf_counter()
    best_differences, highest_round = search.findGoodInputDifferences(cipher_name, scenario, param, search_max_round, output_dir,
        epsilon=epsilon, dual_key=dual_key, T=T, round_step=round_step)
    end = time.perf_counter()
    search_time = round(end - start, 2)

    param_str = '_'.join(str(p) for p in param)
    result_file = os.path.join(output_dir, f"{algorithm}_{param_str}-{max_round}r.txt")
    with open(result_file, "w") as f:
        f.write(f"Algorithm: {algorithm}\n")
        f.write(f"Test param: {param}\n")
        f.write(f"Found {len(best_differences)} {epsilon}-close differences: "
                f"{[hex(x) for x in best_differences]}\n")
        f.write(f"Search round is {highest_round}\n")
        f.write(f"Search time cost: {search_time}s\n")

    print(f"[{algorithm}] param={param}, search_round={highest_round}, "
          f"diffs={len(best_differences)}, search_time={search_time}s")

    # Neural network training phase
    train_results = None
    if train_config is not None:
        import trainer
        print(f"[{algorithm}] Training distinguisher at {max_round:g} rounds "
              f"for {min(len(best_differences), train_config['max_diffs'])} diffs...")
        train_results = trainer.train_for_diffs(
            algorithm, param, best_differences, max_round,
            max_diffs=train_config['max_diffs'],
            num_epochs=train_config['num_epochs'],
            batch_size=train_config['batch_size'],
            data_size=train_config['data_size'])

        # Append training results to file
        with open(result_file, "a") as f:
            f.write(f"\nTraining round: {max_round:g}\n")
            f.write(f"Training epochs: {train_config['num_epochs']}\n")
            for diff_hex, acc, t in train_results:
                f.write(f"  diff={hex(diff_hex)}, acc={acc}, time={t}s\n")
            best_acc = max(acc for _, acc, _ in train_results) if train_results else 0
            f.write(f"Best accuracy: {best_acc}\n")

        print(f"[{algorithm}] Training done. Best acc="
              f"{max(acc for _, acc, _ in train_results) if train_results else 'N/A'}")

    return param, best_differences, highest_round, search_time, train_results


def parse_params(params_str):
    params = json.loads(params_str)
    if not params:
        raise ValueError("Empty parameter list")
    if isinstance(params[0], list):
        return params
    else:
        return [params]


def main():
    parser = argparse.ArgumentParser()

    # Diff search arguments
    parser.add_argument('algorithm', type=str, choices=list(ALGORITHM_CONFIG.keys()), help='Algorithm to search')
    parser.add_argument('max_round', type=float, help='Maximum number of rounds to search')
    parser.add_argument('params', type=str, help='Shift parameter sets as JSON, e.g. "[[4,5],[7,2]]" or "[4,5]" for single set')
    parser.add_argument('--output', type=str, default='results', help='Directory to save results (default: results)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for close differences (default: 0.1)')
    parser.add_argument('--cores', type=int, default=1, help='Number of CPU cores for parallel search (default: 1)')
    
    # Training arguments
    parser.add_argument('--train', action='store_true', help='Enable neural network training after search')
    parser.add_argument('--max-diffs', type=int, default=4, help='Maximum number of diffs to train on (default: 4)')

    args = parser.parse_args()

    param_sets = parse_params(args.params)
    num_params = len(param_sets)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Build training config (None if training is disabled)
    train_config = None
    if args.train:
        train_config = {
            'num_epochs': 10,
            'max_diffs': args.max_diffs,
            'data_size': 10**7,
            'batch_size': 5000,
        }

    print("=" * 70)
    print(f"Cipher:   {args.algorithm}")
    print(f"Max round:   {args.max_round:g}")
    search_mr = get_search_max_round(args.algorithm, args.max_round)
    if search_mr != args.max_round:
        print(f"Search round: {search_mr:g}")
    print(f"Params:      {num_params} set(s): {param_sets}")
    print(f"Epsilon:     {args.epsilon}")
    print(f"CPU cores:   {args.cores}")
    print(f"Output:      {args.output}")
    if train_config:
        print(f"Training:    epochs={train_config['num_epochs']}")
    else:
        print(f"Training:    disabled")
    print("=" * 70)

    task_args = [
        (args.algorithm, param, args.max_round, 'single-key',
         args.output, args.epsilon, train_config)
        for param in param_sets
    ]

    if args.cores > 1 and num_params > 1:
        pool = Pool(processes=min(args.cores, num_params))
        results = pool.map(run_single_param, task_args, chunksize=1)
        pool.close()
        pool.join()
    else:
        results = [run_single_param(t) for t in task_args]

    print("\n" + "=" * 70)
    print("All tasks finished.")
    for param, diffs, highest_round, search_time, train_results in results:
        msg = (f"  param={param}, search_round={highest_round}, "
               f"diffs={len(diffs)}, search_time={search_time}s")
        if train_results:
            best_acc = max(acc for _, acc, _ in train_results)
            total_train_time = round(sum(t for _, _, t in train_results), 2)
            msg += f", best_acc={best_acc}, train_time={total_train_time}s"
        print(msg)


if __name__ == "__main__":
    main()
