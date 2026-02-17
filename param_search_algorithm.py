from copy import copy
from random import randint
from collections import defaultdict
from math import log2
from eval_func import Evaluator
from ciphers.cipher import Cipher

debug_verbose = True

class Search:
    def __init__(self, f: Evaluator, param_num: int, param_size: int, cipher_name: str):
        self.f = f
        self.param_num = param_num
        self.param_size = param_size
        self.cipher_name = cipher_name
    
    def search(self, init_params: list):
        raise NotImplementedError('You need to override this method!')

class greedyOptimizerWithExploration(Search):

    def __init__(self, simple_f: Evaluator, param_num: int, param_size: int, cipher_name: str, N, time, alpha=0.01):
        super().__init__(simple_f, param_num, param_size, cipher_name)
        self.N = N
        self.time = time
        self.alpha = alpha
    
    def search(self, init_params: list, early_stop_val=0.505):
        best_params = copy(init_params)
        guess_params = copy(init_params)
        best_val = val = self.f(Cipher(self.cipher_name, init_params))
        visit_count = defaultdict(int)
        try_num = 0
        l = 0
        if debug_verbose:
            print(f"[Iteration index: 0] Params: {best_params}. Val: {best_val}")
        while try_num < self.N:
            n = (l % self.time) + 1
            l += 1
            for _ in range(n):
                try_num += 1
                visit_count[tuple(guess_params)] += 1

                step = 2 ** (self.time - n) # The step size varies with l
                # If the step size is larger than a certain threshold, random perturbation of the step size ensures that the step size will not always be even
                if step >= 4:
                    if randint(0, 1):
                        step -= 1
                
                index = randint(0, self.param_num - 1)
                direc = 1 if randint(0, 1) else -1

                new_params = copy(guess_params)
                new_params[index] = (new_params[index] + direc * step) % self.param_size

                new_val = self.f(Cipher(self.cipher_name, new_params))
                if debug_verbose:
                    print(f"[Iteration index: {try_num}] Testing params: {new_params}. Val is {new_val}.")
                if new_val < best_val:
                    best_val = new_val
                    best_params = copy(new_params)

                penalty_new = new_val + self.alpha * log2(visit_count[tuple(new_params)] + 1)
                penalty_old = val + self.alpha * log2(visit_count[tuple(guess_params)] + 1)
                if penalty_new < penalty_old:
                    guess_params = copy(new_params)
                    val = new_val

                # Print log
                if debug_verbose:
                    print(f"[Iteration index: {try_num}] Params: {guess_params} and val: {val}. Best params: {best_params} and best val: {best_val}.")
                
                # Determine whether the early stop condition is met
                if best_val < early_stop_val:
                    return best_params, best_val
        return best_params, best_val