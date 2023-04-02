import numpy as np
import itertools


#abaixo 2 funções para o promethee

class Promethee_tensor():


    def __init__(self):
        pass


    def funcao_pref_usual_e_forma_u(self, criterio, atributo_domina, vetor_criteriomax, q):
        #CUIDADO, no código abaixo eu to dizendo que se o atributo é de max ou min independente do critério, então o menor valor é o melhor. Se tiver um atributo de max, tenho que mudar

        if atributo_domina:
            v = [a[0] < (a[1] + q) and 1 or 0 for a in itertools.product(criterio, criterio)]
        else:
            if vetor_criteriomax:
                v = [a[0] > (a[1] + q) and 1 or 0 for a in itertools.product(criterio, criterio)]
            else:
                v = [a[0] < (a[1] + q) and 1 or 0 for a in itertools.product(criterio, criterio)]



        args = [iter(v)] * len(criterio)
        matriz_de_comp = (list(itertools.zip_longest(*args)))

        return np.asarray(matriz_de_comp)





    def run(self, tensor, atributo_domina, vetor_criteriomax, funçao_de_pref, vetor_pesos, alt):

        t = len(tensor)
        matriz_M = np.zeros((tensor.shape[1], tensor.shape[1]))
        for a, matriz in enumerate(tensor):
            for c in range(matriz.shape[1]):
                criterio = matriz[:, c]

                if funçao_de_pref[c][0] == 'usual' or funçao_de_pref[c][0] == 'forma_u':
                    q = funçao_de_pref[c][1]
                    matriz_de_comp = self.funcao_pref_usual_e_forma_u(criterio,atributo_domina[a], vetor_criteriomax[c], q)


                matriz_de_comp = matriz_de_comp * vetor_pesos[a][c]

                matriz_M += matriz_de_comp


        fi_mais = np.array([sum(a) / (len(a) - 1) for a in matriz_M])
        fi_menos = np.array([sum(a) / (len(a) - 1) for a in matriz_M.transpose()])

        fi = fi_mais - fi_menos

        index = [i for i in range(0, len(fi))]
        fi_ordenado = sorted(zip(fi, alt, index), key=lambda x: x[0], reverse=True)
        return fi_ordenado
