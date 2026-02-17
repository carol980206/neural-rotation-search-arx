import sys
import time
import random
import gc
import param_search_algorithm
from eval_func import Evaluator
from neural_evaluator import train_with_good_diff

if __name__ == '__main__':
    cipher_name = sys.argv[1]
    if cipher_name == 'speck32':
        num_rounds = 6
        param_num = 2
        param_size = 16
    elif cipher_name == 'speck48':
        num_rounds = 6
        param_num = 2
        param_size = 24
    elif cipher_name == 'speck64':
        num_rounds = 6
        param_num = 2
        param_size = 32
    elif cipher_name == 'speck96':
        num_rounds = 7
        param_num = 2
        param_size = 48
    elif cipher_name == 'speck128':
        num_rounds = 8
        param_num = 2
        param_size = 64
    elif cipher_name == 'lea':
        num_rounds = 9
        param_num = 3
        param_size = 32
    elif cipher_name == 'chaskey':
        num_rounds = 3.5
        param_num = 6
        param_size = 32
    elif cipher_name == 'siphash':
        num_rounds = 3
        param_num = 6
        param_size = 64
    K = 5
    T = 4
    batch_size = 5000
    batch_num = 2000
    
    # N controls the total number of parameter attempts
    N = 1 << 8
  
    # Create an security evaluator
    simple_evaluator = Evaluator(lambda x: train_with_good_diff(x, 5, num_rounds, T, batch_size, batch_size*batch_num, 'only_diff', False))
    while True:
        gc.collect()
        simple_evaluator.reset()
        # Init param
        init_params = [random.randint(0, param_size - 1) for _ in range(param_num)]
        # Create search process
        searcher = param_search_algorithm.greedyOptimizerWithExploration(simple_evaluator, param_num, param_size, cipher_name, N, K)        
        # Start param search
        start = time.perf_counter()
        best_params, best_val = searcher.search(init_params)
        end = time.perf_counter()
    
        print(f"Best params: {best_params}. Best val: {best_val}.")
        print(f"Search time: {round(end - start, 2)} s.")
        print(f"Simple evaluating times: {simple_evaluator.count}.")
    
        # Output M best params that have been found
        print("Best params list:")
        M = 10
        sorted_values = simple_evaluator.get_best_k_values(M)
        for x in sorted_values:
            print(f"Params: {x[0]}. Value: {x[1]} ")

        output_file = open(f"./{cipher_name}_search_results-{num_rounds}r.txt", mode='a')
        output_file.write("New search trial:\n")
        for x in sorted_values:
            output_file.write(f"Params: {x[0]}. Value: {x[1]} \n")
        output_file.write('\n')
        output_file.close()