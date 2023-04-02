import numpy as np
import itertools
from scipy import stats


#abaixo 2 funções para o promethee

class Atributos():


    def __init__(self):
        pass


    def run(self, tensor_sinais):

        tensor_atributos = []
        mean = np.mean(tensor_sinais, axis=0)

        tensor_atributos.append(mean)

        ca = []
        tensor_sinais = np.array(tensor_sinais)

        for i in tensor_sinais.T:

            aux = []
            for j in i:
                slope, intercept, r_value, p_value, std_err = stats.linregress((np.arange(len(j)), j))
                aux.append(slope)
            ca.append(aux)
        coef_angular = np.array(ca).T

        tensor_atributos.append(coef_angular)

        cv = stats.variation(tensor_sinais, axis=0)

        tensor_atributos.append(cv)
        return tensor_atributos

