import json
import numpy as np
import math




def get_signals(n_a, N, variancia_ruido, nome_da_pasta, nome_arquivo, n_iterations):
    # 𝓁 = 1000 times, and we averaged the results of  these 𝓁 simulations.

    m_d_1 = []
    m_d_2 = []
    m_d_3 = []
    m_r = []

    for iteracao in range(n_iterations):

        # Signals criterion 1
        t = np.arange(1, N + 1)
        a = np.random.uniform(2, 20, n_a)
        b = np.random.uniform(0.5, 0.8, n_a)
        d_1 = []
        for i in range(len(a)):
            d = a[i] + b[i] * t
            d_1.append((d).tolist())
        m_d_1.append(d_1)

        # Signals criterion 2
        n = np.arange(1, N + 1)
        a = np.random.uniform(2, 20, n_a)
        freq = np.random.uniform(0.7, 0.75, n_a) * math.pi
        d_2 = []
        for i in range(len(freq)):
            d = a[i] + np.sin(freq[i] * n)
            d_2.append((d).tolist())
        m_d_2.append(d_2)

        # Signals criterion 3

        d_3 = []
        a = np.random.uniform(2, 20, n_a)
        for i in range(n_a):
            d = a[i] + (-1) ** t
            d_3.append((d).tolist())

        m_d_3.append(d_3)



        # The white noise vector:
        r = np.random.normal(0, math.sqrt(variancia_ruido), N)
        m_r.append(r.tolist())

    fp = open("%s/%s" % (nome_da_pasta, nome_arquivo), "w")
    dict_dados = {
        'd_1': m_d_1,
        'd_2': m_d_2,
        'd_3': m_d_3,
        'r': m_r,
    }

    fp.write(json.dumps(dict_dados) + "\n")
