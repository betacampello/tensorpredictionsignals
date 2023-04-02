import numpy as np
import matplotlib.pylab as plt
import sys
import json
import scipy
from scipy import signal
from numpy.linalg import inv
#from padasip.filters.base_filter import AdaptiveFilter


# https://matousc89.github.io/padasip/sources/filters/rls.html

class FilterRLS_beta():


    def __init__(self):
        pass



    def run(self, d, x, passo, N, lamb, gama, parametros):


        N = len(d)


        try:
            x = np.array(x)
            d = np.array(d)
        except:
            raise ValueError('Impossible to convert x or d to a numpy array')

        # create empty arrays
        S = gama * np.identity(parametros)
        y = np.zeros(N -(passo-1))
        e = np.zeros(N -(passo-1))
        epi = np.zeros(N -(passo-1))
        self.w_history = np.zeros((N-(passo-1), parametros))
        self.w = np.zeros(parametros)
        d = d[passo-1:len(d)]

        # adaptation loop
        # algoritmo pag 213(218) -  Adaptive Filtering
        # Algorithms and Practical Implementation

        for k in range(0, N -(passo-1)):
            self.w_history[k, :] = self.w

            y[k] = np.dot(self.w.T, x[k])
            e[k] = d[k] - np.dot(x[k].T, self.w)




            a = np.dot(np.transpose(x[k]), S)

            b = 1 / (lamb + np.dot(a, x[k]))

            g_n = np.dot(S, x[k]) * b

            c = (1 / lamb) * S


            aux = np.zeros((parametros, parametros))
            for aa in range(len(g_n)):
                for bb in range(len(x[k])):
                    aux[aa][bb] = g_n[aa] * x[k][bb]


            S = c - np.dot(aux, c)
            aux2 = g_n * e[k]
            self.w = self.w + aux2



        predicao = np.dot(self.w, x[-1])
        #print('x-1',x[-1])
        #print('predicao rls a passo %d: %.5f'%(passo, predicao))

        return y, e, self.w_history, predicao



