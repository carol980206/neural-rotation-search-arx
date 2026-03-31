from train_ciphers.speck_train import SpeckTrain
from train_ciphers.lea_train import LEATrain
from train_ciphers.chaskey_train import ChaskeyTrain
from train_ciphers.siphash_train import SiphashTrain


def get_train_cipher(algorithm, param):
    if algorithm.startswith('speck'):
        return SpeckTrain(algorithm, alpha=param[0], beta=param[1])
    elif algorithm == 'lea':
        return LEATrain(a=param[0], b=param[1], c=param[2])
    elif algorithm == 'chaskey':
        return ChaskeyTrain(c=param)
    elif algorithm == 'siphash':
        return SiphashTrain(c=param)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
