import pandas as pd
from rls import FilterRLS_beta
import numpy as np
import sys
#from promethee import Promethee
from kendal_tau import Kendal
from numpy import sqrt
import random
import matplotlib.pylab as plt
from matplotlib import pyplot
import pandas as pd
import collections
from calc_atributos_sinal import Atributos
from promethee_para_tensores import Promethee_tensor
from topsis import Topsis
from lms_teste import LMS

np.set_printoptions(suppress=True)

current = 6
filt_paramet = 2
v_passo = [1, 2, 3, 4, 5,6]
n_iteracao_smaa = 10000
# LMS:
mu_lms = 0.3

nome_da_pasta = 'data'
nome_arquivo = 'paises_selecionados_teste2.xlsx'

a = 5
c = 3
q = a * c

lamb = 0.9  # Fator de esquecimento do RLS
gama = 0.1  # Para inicializar a matriz R do RLS

crit_max = [True, False, False]
atributo_domina = [False, False, True]

funçao_de_pref = [('usual', 0), ('usual', 0), ('usual', 0)]

v_p = [0.2, 0.4, 0.4]
v_p = [0.2, 0.5, 0.3]
# v_p = [0.25, 0.25, 0.5]
# v_p = [0.7, 0.2, 0.1]
v_p = [1 / 3, 1 / 3, 1 / 3]

alt = ['a%d' % (d + 1) for d in range(a)]  # This vector is just to name the variables as a_1...a_n

# nome_arquivo = '4paises_criterios_com_inflacao.xlsx'
df = pd.read_excel("%s/%s" % (nome_da_pasta, nome_arquivo))
matriz_classica = np.zeros(q)

m_p_RLS = []
m_p_LMS = []
m_p_i = []
tensor_passado_completo = []
tensor_passado_janelado = []

matriz_media_des_erro = []
matriz_intervalo_predicao = []
ponto_dento = 0
v_d_futuro = []
v_intervalo_de_confianca = []
v_valor_real_dentro_intervalo = []
v_R_predicao = []
v_L_predicao = []
v_desvio_padrao = []

for passo in v_passo:
    matriz_pred_RLS = np.zeros(q)
    matriz_pred_LMS = np.zeros(q)
    matriz_pred_ideal = np.zeros(q)

    for i in range(q):
        if i == 1 or i == 4 or i == 7 or i == 10 or i == 13 or i == 16:
            lamb = 0.99
        else:
            if passo == 3:
                lamb = 0.9
            else:
                lamb = 0.9

        d_completo = df.loc[i]



        d_completo = np.array(d_completo)
        # Montar as matrizes com valores atuais (current value) e com valores "futuros"
        if passo == 1:
            d_serie_temporal_passado_completa = d_completo[:-current]
            d_serie_temporal_passado_janelada = d_completo[28:-current]

            tensor_passado_completo.append(d_serie_temporal_passado_completa)
            tensor_passado_janelado.append(d_serie_temporal_passado_janelada)

        d_futuro = d_completo[-1 - current + passo]
        d_atual = d_completo[-1 - current]
        v_d_futuro.append(d_futuro)
        matriz_classica[i] = d_atual
        matriz_pred_ideal[i] = d_futuro

        # sinal desejado d
        d = d_completo[1:len(d_completo) - current]


        # <editor-fold desc="obter o x, que é o que considero o sinal de entrada do lmsn e rls">
        N = len(d)
        x = [[0] * filt_paramet]
        aux = np.concatenate((np.zeros(filt_paramet), d))
        for k in range(filt_paramet, len(d) + filt_paramet):
            x.append([aux[k - i] for i in range(0, filt_paramet)])
        # </editor-fold>

        # Filtrando com RLS

        Rf = FilterRLS_beta()
        Ry, Re, Rw, Rpredicao = Rf.run(d, x, passo, N, lamb, gama, filt_paramet)

        anos = np.arange(1980, 2013)

        # Filtrando com LMS
        f = LMS()
        Ly, Le, Lw, Lpredicao = f.run(d, x, passo, N, mu_lms, filt_paramet)


        re_2 = np.square(Re[3:])
        sum_re_2 = np.sum(re_2)
        var_erro = (1 / (len(re_2) - 1)) * sum_re_2
        desvio_erro = sqrt(var_erro)
        amplitude = 1.96 * desvio_erro
        intervalo_predicao = [Rpredicao - amplitude, Rpredicao + amplitude]
        # print('intervalo', intervalo_predicao)
        # print('valor de pred', Rpredicao)
        # print('valor real', d_futuro)
        # print(Rpredicao)
        # print(desvio_erro)

        # matriz_media_des_erro.append([Rpredicao, desvio_erro])
        # matriz_intervalo_predicao.append([Rpredicao - amplitude, Rpredicao + amplitude])
        v_R_predicao.append(Rpredicao)
        v_L_predicao.append(Lpredicao)
        v_desvio_padrao.append(desvio_erro)
        # print(matriz_media_des_erro)

        matriz_pred_RLS[i] = Rpredicao
        matriz_pred_LMS[i] = Lpredicao

    m_c = matriz_classica.reshape((a, c))
    m_p_i.append(matriz_pred_ideal.reshape((a, c)))
    ideal = matriz_pred_ideal.reshape((a, c))
    pred = matriz_pred_RLS.reshape((a, c))
    m_p_RLS.append(matriz_pred_RLS.reshape((a, c)))
    m_p_LMS.append(matriz_pred_LMS.reshape((a, c)))

    # matriz_media_des_erro = np.array(matriz_media_des_erro).reshape((a, c,2))
    # matriz_intervalo_predicao = np.array(matriz_intervalo_predicao).reshape((a, c, 2))

    # <editor-fold desc="rankin_c">
    #C_promethee = Promethee()
    #c_v_ordenado = C_promethee.run(m_c, crit_max, funçao_de_pref, v_p, alt)

    #c_ranking = []
    #for k in range(len(c_v_ordenado)):
    #    c_ranking.append(c_v_ordenado[k][2])
    # </editor-fold>
    #print('ranking curre', c_ranking)

    # <editor-fold desc="ranking_pi">
    #P_promethee = Promethee()
    #p_v_ordenado = P_promethee.run(ideal, crit_max, funçao_de_pref, v_p, alt)
    # print(p_v_ordenado)

    #p_ranking = []
    #for k in range(len(p_v_ordenado)):
    #    p_ranking.append(p_v_ordenado[k][2])
    # </editor-fold>

    #print('ranking ideal', p_ranking)

    # <editor-fold desc="kendal1">
    # To compare the ranking with the ideal decision matrix
    #kendal = Kendal()
    #tau1 = kendal.run(c_ranking, p_ranking)
    # </editor-fold>
    # print('dif ideal e atual')
    # print(tau1)

    # <editor-fold desc="ranking_p">
    #PR_promethee = Promethee()
    #pr_v_ordenado = PR_promethee.run(pred, crit_max, funçao_de_pref, v_p, alt)
    # print(pr_v_ordenado)

    #pr_ranking = []
    #for k in range(len(pr_v_ordenado)):
    #    pr_ranking.append(pr_v_ordenado[k][2])
    # </editor-fold>

    #print('ranking estim', pr_ranking)
    # print('ranking novon', p2_ranking)

    # <editor-fold desc="kendal2">
    # To compare the ranking with the ideal decision matrix
    #kendal = Kendal()
    #tau2 = kendal.run(p_ranking, pr_ranking)
    # </editor-fold>
    #print('dif ideal e predição')
    #print(tau2)

cal_atributos2 = Atributos()
tensor_atributos_valores_ideais = cal_atributos2.run(m_p_i)


v_p = [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
C_promethee2 = Promethee_tensor()
c_v_ordenado2 = C_promethee2.run(np.array(tensor_atributos_valores_ideais), atributo_domina, crit_max, funçao_de_pref, v_p, alt)

print('ord promethee ideal', c_v_ordenado2)

cal_atributos = Atributos()
tensor_atributos_valores_RLSpredicao = cal_atributos.run(m_p_RLS)
tensor_atributos_valores_LMSpredicao = cal_atributos.run(m_p_LMS)


#PROMETHEE II RLS
C_promethee = Promethee_tensor()
c_v_ordenado = C_promethee.run(np.array(tensor_atributos_valores_RLSpredicao), atributo_domina, crit_max, funçao_de_pref, v_p, alt)
print('ord promethee predicao RLS', c_v_ordenado)


#PROMETHEE II LMS
C_promethee_lms = Promethee_tensor()
c_v_ordenado_lms = C_promethee_lms.run(np.array(tensor_atributos_valores_LMSpredicao), atributo_domina, crit_max, funçao_de_pref, v_p, alt)

print('ord promethee predicao LMS', c_v_ordenado_lms)



# TOPSIS extension:



vp_t = [1/3,1/3,1/3]
vp = [1/3,1/3,1/3]
atri_minmax = [False, False, True]

#TOPSIS RLS

topsis_RLS = Topsis()
# O tensor do espaço de atributos eu pondero tanto os critérios como as séries temporais
esp_atributo_ponderado_terceira_dimensão = topsis_RLS.ponderar_terceira_dimensao(tensor_atributos_valores_RLSpredicao, vp_t)

#2. Já normalizei os atributos, mas esse seria o passo 2

esp_atributo_ponderado = topsis_RLS.ponderar(esp_atributo_ponderado_terceira_dimensão, vp)

crit_max = [True]*10
# Identifico o vetor ideal e o nadir
ideal_esp_atributo_ponderado, nadir_esp_atributo_ponderado = topsis_RLS.ideal_nadir(esp_atributo_ponderado, crit_max, atri_minmax)

# Calculo as distâncias entre o valores dos critérios e os valores ideais e nadir
esp_atributo_d_ideal, esp_atributo_d_nadir = topsis_RLS.distancia_id_nad(esp_atributo_ponderado, ideal_esp_atributo_ponderado, nadir_esp_atributo_ponderado)

# Calculo v e ordeno ele
esp_atributo_v_ordenado = topsis_RLS.calc_e_ordenamento_v(esp_atributo_d_ideal, esp_atributo_d_nadir, alt)
print('ord topsis RLS', esp_atributo_v_ordenado)


#TOPSIS LMS
topsis_LMS = Topsis()

# O tensor do espaço de atributos eu pondero tanto os critérios como as séries temporais
esp_atributo_ponderado_terceira_dimensão = topsis_LMS.ponderar_terceira_dimensao(tensor_atributos_valores_LMSpredicao, vp_t)

#2. Já normalizei os atributos, mas esse seria o passo 2

esp_atributo_ponderado = topsis_LMS.ponderar(esp_atributo_ponderado_terceira_dimensão, vp)

crit_max = [True]*10
# Identifico o vetor ideal e o nadir
ideal_esp_atributo_ponderado, nadir_esp_atributo_ponderado = topsis_LMS.ideal_nadir(esp_atributo_ponderado, crit_max, atri_minmax)

# Calculo as distâncias entre o valores dos critérios e os valores ideais e nadir
esp_atributo_d_ideal, esp_atributo_d_nadir = topsis_LMS.distancia_id_nad(esp_atributo_ponderado, ideal_esp_atributo_ponderado, nadir_esp_atributo_ponderado)

# Calculo v e ordeno ele
esp_atributo_v_ordenado = topsis_LMS.calc_e_ordenamento_v(esp_atributo_d_ideal, esp_atributo_d_nadir, alt)
print('ord topsis LMS', esp_atributo_v_ordenado)




#TOPSIS BENCHMARCK

topsis_IDEAL = Topsis()

# O tensor do espaço de atributos eu pondero tanto os critérios como as séries temporais
esp_atributo_ponderado_terceira_dimensão = topsis_IDEAL.ponderar_terceira_dimensao(tensor_atributos_valores_ideais, vp_t)

#2. Já normalizei os atributos, mas esse seria o passo 2

esp_atributo_ponderado = topsis_IDEAL.ponderar(esp_atributo_ponderado_terceira_dimensão, vp)

crit_max = [True]*10
# Identifico o vetor ideal e o nadir
ideal_esp_atributo_ponderado, nadir_esp_atributo_ponderado = topsis_IDEAL.ideal_nadir(esp_atributo_ponderado, crit_max, atri_minmax)

# Calculo as distâncias entre o valores dos critérios e os valores ideais e nadir
esp_atributo_d_ideal, esp_atributo_d_nadir = topsis_IDEAL.distancia_id_nad(esp_atributo_ponderado, ideal_esp_atributo_ponderado, nadir_esp_atributo_ponderado)

# Calculo v e ordeno ele
esp_atributo_v_ordenado = topsis_IDEAL.calc_e_ordenamento_v(esp_atributo_d_ideal, esp_atributo_d_nadir, alt)
print('ord topsis ideal', esp_atributo_v_ordenado)