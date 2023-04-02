import numpy as np
from scipy.spatial import distance
from numpy import linalg as LA
from scipy import stats



tensor = np.array([
                [[4, 3],
                [9, 2]],

                [[2, 4],
                 [1, 2]]
                ])

print(tensor)


def trans_espaco_atributos(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    var = np.var(X, axis=0)
    cv = stats.variation(X, axis=0)

    tensor_atributos = np.array([mean, std, var, cv])

    return tensor_atributos

# Não normalizo, primeiro pego as características
tensor_atributos = trans_espaco_atributos(tensor)
print(tensor_atributos)

vetor_pesos = [0.3, 0.7]

tensor_atr_ponderado = [(np.multiply(i, vetor_pesos)).tolist() for i in tensor_atributos]

print(np.array(tensor_atr_ponderado))
