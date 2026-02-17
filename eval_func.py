from ciphers.cipher import Cipher

class Evaluator(object):
    def __init__(self, f):
        self.eval_value_dict = dict()
        self.count = 0
        self.f = f

    def __call__(self, cipher: Cipher):
        self.count += 1
        tweakable_params = tuple(cipher.tweakable_params)
        if tweakable_params not in self.eval_value_dict:
            self.eval_value_dict[tweakable_params] = self.f(cipher)
        return self.eval_value_dict[tweakable_params]

    def reset(self):
        self.count = 0
        self.eval_value_dict.clear()

    def get_best_k_values(self, k):
        sorted_values = sorted(self.eval_value_dict.items(), key = lambda x : x[1])
        return sorted_values[:k]