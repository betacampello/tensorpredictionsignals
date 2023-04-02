import numpy as np
import matplotlib.pylab as plt
import sys
import json
import scipy
from scipy import signal
from numpy.linalg import inv
from numpy import linalg as LA


# https://matousc89.github.io/padasip/sources/filters/rls.html

class LMS():


    def __init__(self):
        pass



    def run(self, d, x, passo, N, mu,  parametros):

        # Initialization
        x = np.array(x)
        d = np.array(d)
        self.mu = mu


        w = np.zeros(parametros)

        e = np.zeros(N -(passo-1))
        w_history = np.zeros((N - (passo - 1), parametros))
        y = np.zeros(N - (passo - 1))
        d = d[passo - 1:len(d)]

        # Calculo do valor máximo de mu:
        aux = np.dot(x, np.transpose(x))
        eigval = LA.eig(aux)
        lambda_max = np.max(np.real(eigval[0]))
        maior_lambda = 2/lambda_max

        max_range = N -(passo-1)
        for k in range(0, max_range):


            w_history[k, :] = w

            y[k] = np.dot(w.T, x[k])

            e[k] = d[k] - np.dot(x[k].T, w)


            w = w + (mu * x[k] * e[k])

            #print('k', k)
            #print('e/d', e[k]/d[k])
            #print('d', d[k])
            #print('w', w)
            #print('mu', mu)

            #if k > (0.5*max_range) and mu > maior_lambda:
            #    mu = maior_lambda
            if np.abs(e[k]) > np.abs(d[k]):
                mu = maior_lambda
            else:
                mu = self.mu

            #print('mu', mu)





        predicao = np.dot(w, x[-1])


        return y, e, w_history, predicao