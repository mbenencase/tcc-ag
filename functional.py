import time
import pandapower as pp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def initialize_pes(pes):

    if len(pes.bus) == 14:

        pes.bus['min_vm_pu'] = 0.95

        pes.bus['max_vm_pu'] = 1.05

        pp.runpp(pes, init='results', tolerance_mva=1e-6, trafo_model='pi')

    if len(pes.bus) == 30:

        pes.bus['min_vm_pu'] = 0.95

        pes.bus['max_vm_pu'] = 1.05

        pes.gen['max_q_mvar'] = np.array([50, 40, 40, 24, 24])

        pes.gen['min_q_mvar'] = np.array([-40, -40, -10, -6, -6])

        pes.ext_grid['max_q_mvar'] = 10

        pes.ext_grid['min_q_mvar'] = 0

        pp.runpp(pes, init='results', tolerance_mva=1e-6)

    if len(pes.bus) == 118:

        pes.bus['min_vm_pu'] = 0.94

        pes.bus['max_vm_pu'] = 1.06

        pes.ext_grid['va_degree'] = 0

        pp.runpp(pes, init='results', tolerance_mva=1e-6, trafo_model='pi')

    if len(pes.bus) == 300:

        pes.bus['min_vm_pu'] = 0.9
        pes.bus['max_vm_pu'] = 1.1

        pp.runpp(pes, init='results', tolerance_mva=1e-6, trafo_model='pi')

    voltages_init = pes.gen['vm_pu'].to_numpy()

    tap_pos = pes.trafo[~pd.isnull(
        pes.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)

    tap_neutral = pes.trafo[~pd.isnull(
        pes.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)

    tap_step_percent = pes.trafo[~pd.isnull(
        pes.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)

    valor_pu_tap = (tap_pos-tap_neutral)*(tap_step_percent/100) + 1

    valor_bshunt = (pes.shunt['q_mvar']/(-100)).to_numpy()

    zeros = np.array([0, 0, 0, 0, 0, 0])

    valor_inicial = np.expand_dims(np.concatenate(
        (voltages_init, valor_pu_tap, valor_bshunt, zeros), axis=None), 0)

    return valor_inicial


def conductance_matrix(pes, relatorio=True):
    pes.line = pes.line.sort_index()

    pes.bus = pes.bus.sort_index()

    vbus = pes.bus.vn_kv.to_numpy(dtype=np.float64)

    zbase = np.power(np.multiply(vbus, 1000), 2)/100e6

    # Inicializa Matriz Zerada

    matriz_z = np.zeros((9, len(pes.line.index.ravel())), dtype=np.float64)

    matriz_g = np.zeros(
        (pes.bus.name.count(), pes.bus.name.count()), dtype=np.float64)

    g = np.zeros(len(pes.line.index.ravel()), dtype=np.float64)

    # Pega Valores de Barra Origem e Destino das Linhas

    matriz_z[0, :] = pes.line.from_bus

    matriz_z[1, :] = pes.line.to_bus

    for i in range(len(pes.line.index.ravel())):

        matriz_z[2, i] = pes.line.r_ohm_per_km[i]/zbase[int(matriz_z[0, i])]
        matriz_z[3, i] = pes.line.x_ohm_per_km[i]/zbase[int(matriz_z[0, i])]

    # Calcula Condutâncias

    g = np.array(np.divide(matriz_z[2, :], np.power(
        matriz_z[2, :], 2)+np.power(matriz_z[3], 2)))
    z = np.sqrt(np.power(matriz_z[2, :], 2) + np.power(matriz_z[3, :], 2))
    b = np.array(np.divide(matriz_z[3, :], np.power(
        matriz_z[2, :], 2)+np.power(matriz_z[3], 2)))
    matriz_z[4, :] = g

    vo = []
    vd = []
    to = []
    td = []

    for bus in matriz_z[0, :]:

        vo.append(pes.res_bus['vm_pu'][pes.res_bus.index ==
                  bus].to_numpy(dtype=np.float64).item())
        to.append(pes.res_bus['va_degree'][pes.res_bus.index == bus].to_numpy(
            dtype=np.float64).item())

    for bus in matriz_z[1, :]:

        vd.append(pes.res_bus['vm_pu'][pes.res_bus.index ==
                  bus].to_numpy(dtype=np.float64).item())
        td.append(pes.res_bus['va_degree'][pes.res_bus.index == bus].to_numpy(
            dtype=np.float64).item())

    matriz_z[5, :] = vo
    matriz_z[6, :] = to
    matriz_z[7, :] = vd
    matriz_z[8, :] = td

    # Gera Matriz

    for posicao in range(len(pes.line.index.ravel())):

        matriz_g[matriz_z[0, posicao].astype(
            np.int32), matriz_z[1, posicao].astype(np.int32)] = g[posicao]

    if relatorio == True:

        tabela = np.zeros((len(pes.line.index.ravel()), 7))
        tabela[:, 0] = matriz_z[0, :]
        tabela[:, 1] = matriz_z[1, :]
        tabela[:, 2] = matriz_z[2, :]
        tabela[:, 3] = matriz_z[3, :]
        tabela[:, 4] = z
        tabela[:, 5] = g
        tabela[:, 6] = b

        # table = tabulate.tabulate(tabela, headers=[
                                  # 'Barra de Origem', 'Barra de Destino', 'R (pu)', 'Xl (pu)', 'Z (pu)', 'G (pu)', 'B (pu)'], tablefmt="psql")
        # print(table)

        if len(pes.bus) == 14:

            sns.heatmap(matriz_g+matriz_g.T, annot=True, fmt='.6g', cmap='jet')
            plt.xlabel('Barra Origem')
            plt.ylabel('Barra Destino')
            plt.title('Matriz de Condutâncias de Linha Completa')

    if relatorio == False:

        return matriz_g, matriz_z


def get_vbus_data(pes, relatorio=True):
    '''

    Coleta os Dados de Tensões e Limites Superiores e Inferiores das Barras do Sistema.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    Parâmetros
    ----------
    pes : sistema elétrico de potência carregado pelo pandapower.
    relatorio : caso relatorio = True, retorna relatório informando, tensões, ângulos e limites.
                caso relatorio = False, retorna apenas as tensões, ângulos e limtes

    Retorno
    ----------
    vbus : vetor de tensões [pu] das barras em ordem crescente do número da barra
    theta : vetor de ângulo de tensões [°]
    v_lim_superior : 

    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    '''

    pes.res_bus = pes.res_bus.sort_index()

    pes.bus = pes.bus.sort_index()

    vbus = pes.res_bus['vm_pu'].to_numpy(dtype=np.float64)

    theta = pes.res_bus['va_degree'].to_numpy(dtype=np.float64)

    v_lim_superior = pes.bus["max_vm_pu"].to_numpy(dtype=np.float32)

    v_lim_inferior = pes.bus["min_vm_pu"].to_numpy(dtype=np.float32)

    if relatorio == True:

        tabela = np.zeros((len(vbus), 4))
        tabela[:, 0] = vbus
        tabela[:, 1] = theta
        tabela[:, 2] = v_lim_inferior
        tabela[:, 3] = v_lim_superior

        # table = tabulate.tabulate(tabela, headers=[
        #                           'Tensões nas Barras (pu)', 'Ângulos das Barras (°)', 'Limites Inferiores', 'Limites Superiores'], tablefmt="psql")
        # print(table)

        sns.scatterplot(x=np.arange(0, len(vbus), 1), y=vbus,
                        color='blue', label='Módulo da Tensão', s=75)
        sns.lineplot(x=np.arange(0, len(vbus), 1), y=v_lim_superior,
                     color='red', label='Limite Superior', alpha=0.5)
        sns.lineplot(x=np.arange(0, len(vbus), 1), y=v_lim_inferior,
                     color='orange', label='Limite Inferior', alpha=0.5)
        plt.title('Módulo da Tensão por Barra do Sistema')
        plt.xlabel('Barra do Sistema')
        plt.ylabel('Tensão [pu]')
        plt.legend(loc='best')
        plt.savefig("tensao.png")

        plt.figure(figsize=(16, 10))
        sns.scatterplot(x=np.arange(0, len(theta), 1), y=theta,
                        color='green', label='Ângulo da Tensão', s=75)
        plt.title('Ângulo da Tensão por Barra do Sistema')
        plt.xlabel('Barra do Sistema')
        plt.ylabel('Theta [°]')
        plt.legend(loc='best')
        plt.savefig("angulos.png")

    return vbus, theta, v_lim_superior, v_lim_inferior

    ##################################################################################################################################################################################


def get_generator_data(pes, relatorio=True):
    '''

    Coleta os Dados de Tensões, Potências Ativa e Reativa e Seus Respectivos Limites Superiores e Inferiores de geração.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    Parâmetros
    ----------
    pes : sistema elétrico de potência carregado pelo pandapower.
    relatorio : caso relatorio = True, retorna relatório informando, limites, potências e gráficos.
                caso relatorio = False, retorna apenas as tensões, ângulos, potências e limites.

    Retorno
    ----------
    vgen : vetor de tensões [pu] das barras de geração
    theta : vetor de ângulo de tensões [°] das barras de geração
    p_lim_superior : Limite Superior de Potência Ativa (pu)
    p_lim_inferior : Limite Inferior de Potência Ativa (pu)
    q_lim_superior : Limite Superior de Potência Reativa (pu)
    q_lim_inferior : Limite Inferior de Potência Ativa (pu)

    Observações:
    - - - - - - -

    Potência Aparente de Base : 100 MVA
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    '''

    pes.res_gen = pes.res_gen.sort_index()

    pes.gen = pes.gen.sort_index()

    vgen = pes.res_gen['vm_pu'].to_numpy(dtype=np.float64)

    barra = pes.gen['bus'].to_numpy(dtype=np.float64)

    thetagen = pes.res_gen['va_degree'].to_numpy(dtype=np.float64)

    pgen = pes.res_gen['p_mw'].to_numpy(dtype=np.float64)/100

    qgen = pes.res_gen['q_mvar'].to_numpy(dtype=np.float64)/100

    p_lim_superior = pes.gen["max_p_mw"].to_numpy(dtype=np.float32)/100

    p_lim_inferior = pes.gen["min_p_mw"].to_numpy(dtype=np.float32)/100

    q_lim_superior = pes.gen["max_q_mvar"].to_numpy(dtype=np.float32)/100

    q_lim_inferior = pes.gen["min_q_mvar"].to_numpy(dtype=np.float32)/100

    if relatorio == True:

        tabela = np.zeros((len(vgen), 6))
        tabela[:, 0] = pgen
        tabela[:, 1] = p_lim_superior
        tabela[:, 2] = p_lim_inferior
        tabela[:, 3] = qgen
        tabela[:, 4] = q_lim_superior
        tabela[:, 5] = q_lim_inferior

        # table = tabulate.tabulate(tabela, headers=[
        #                           'P (pu)', 'P Lim. Sup. (pu)', 'P Lim. Inf. (pu)', 'Q (pu)', 'Q Lim. Sup. (pu)', 'Q Lim. Inf. (pu)'], tablefmt="psql")
        # print(table)

        sns.scatterplot(x=barra, y=qgen, color='blue',
                        label='Potência Gerada', s=75)
        sns.lineplot(x=barra, y=q_lim_superior, color='red',
                     label='Limite Superior', alpha=0.5)
        sns.lineplot(x=barra, y=q_lim_inferior, color='orange',
                     label='Limite Inferior', alpha=0.5)
        plt.title('Potência Reativa Gerada')
        plt.xlabel('Barra do Sistema')
        plt.ylabel('Potência Reativa (pu)')
        plt.legend(loc='best')

    if relatorio == False:

        return vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior, barra

    ################################################################################################################################################################################################


def func_objetivo(vbarra, theta, condutancias, matriz_z, relatorio=True):
    '''

    Calcula as perdas nas linhas de transmissão de acordo com as tensões, ângulos das barras e condutâncias de linha.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    Parâmetros
    ----------
    vbarra : tensão da barra.
    theta : ângulo da barra.
    condutancias : matriz de condutâncias de linha (triângulo superior)

    caso relatorio = True, retorna relatório informando a matriz de perdas de linha e as perdas totais.
    caso relatorio = False, retorna apenas as perda em pu.
    Retorno
    ----------

    perdas : perdas de potência ativa em pu.


    Observações:
    - - - - - - -

    Potência Aparente de Base : 100 MVA
     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    '''

    matriz_v = np.zeros((len(vbarra), len(vbarra)), dtype=np.float64)

    matriz_theta = np.zeros((len(theta), len(theta)), dtype=np.float64)

    for barra in range(len(vbarra)):

        matriz_v[:, barra] = vbarra
        matriz_theta[:, barra] = theta

    soma_v = np.power(matriz_v, 2) + np.power(matriz_v.T, 2)

    subtrai_theta = matriz_theta - matriz_theta.T

    cosenotheta = np.cos(np.radians(subtrai_theta))

    produto = 2 * np.multiply(np.multiply(matriz_v, matriz_v.T), cosenotheta)

    matriz_perdas = np.multiply(condutancias, soma_v-produto)

    perdas = np.multiply(matriz_z[4, :], np.power(matriz_z[5, :], 2)+np.power(matriz_z[7, :], 2)-2*np.multiply(
        np.multiply(matriz_z[5, :], matriz_z[7, :]), np.cos(np.radians(matriz_z[8, :]-matriz_z[6, :]))))
    perdas = np.sum(perdas)

    if relatorio == True:

        tabela = np.zeros((1, 2))
        tabela[:, 0] = perdas
        tabela[:, 1] = perdas*100
        # table = tabulate.tabulate(tabela, headers=[
        #                           'Perdas Totais Nas Linhas (pu)', 'Perdas Totais Nas Linhas (MW)'], tablefmt="psql")
        # print(table)

        if len(vbarra) == 14:
            plt.figure(figsize=(18, 10))
            sns.heatmap(100*(matriz_perdas+matriz_perdas.T),
                        annot=True, cmap="jet")
            plt.xlabel('Barra Origem')
            plt.ylabel('Barra Destino')
            plt.title('Matriz de Perdas de Linha Completa [MW]')

    else:

        return perdas

    ################################################################################################################################################################################################


def pen_tensao(vbus, limite_sup, limite_inf, relatorio=True):
    """    
    Calcula a parcela de penalização pura (sem constante de multiplicação) referente a violação dos limites de tensão.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    Parâmetros
    ----------   
    vbus : tensões das barras do sistema elétrico.
    limite_sup : limite superior das tensões das barras do sistema elétrico.
    limite_inf : limite inferior das tensões das barras do sistema elétrico.

    limite_sup : tensões
    relatorio : caso relatorio = True, retorna penalização e nº de violações 
                caso relatorio = False, retorna apenas o valor de penalização.
    Retorno
    -------    
    penalizacao: somatório da diferença ao quadradado das tensões que ultrapassaram os limites inferiores ou superiores.

    Observações:
    ------------

    ...

    """

    inferior = vbus - limite_inf
    superior = limite_sup - vbus
    penalizacao = np.sum(
        np.abs(superior[superior < 0]))+np.sum(np.abs(inferior[inferior < 0]))

    if relatorio == True:

        print('Penalização de Tensão:\n')
        print(penalizacao, '\n')
        print('Número de Violações:\n')
        print(len(inferior[inferior < 0])+len(superior[superior < 0]))

    else:

        return penalizacao

 ################################################################################################################################################################################################


def pen_ger_reativo(q, limite_sup, limite_inf, pes, relatorio=True):
    """    
    Calcula a parcela de penalização pura (sem constante de multiplicação) referente a violação dos limites de geração de reativos.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    Parâmetros
    ----------   
    q : potências reativas das barras de controle de reativo do sistema elétrico.
    limite_sup : limite superior das potências reativas das barras de controle de reativo do sistema elétrico.
    limite_inf : limite superior das potências reativas das barras de controle de reativo do sistema elétrico.

    limite_sup : tensões
    relatorio : caso relatorio = True, retorna penalização e nº de violações 
                caso relatorio = False, retorna apenas o valor de penalização.
    Retorno
    -------    
    penalização: somatório da diferença ao quadradado das potências reativas que ultrapassaram os limites inferiores ou superiores.

    Observações:
    ------------

    ...

    """

    inferior = limite_inf - q
    superior = limite_sup - q

    ext_sup = pes.ext_grid['max_q_mvar'].to_numpy()
    ext_inf = pes.ext_grid['min_q_mvar'].to_numpy()

    qext = pes.res_ext_grid['q_mvar'].to_numpy()

    inferiorext = ext_inf - qext
    superiorext = ext_sup - qext

    penalizacaoext = np.sum(np.abs(
        superiorext[superiorext < 0]))+np.sum(np.abs(inferiorext[inferiorext > 0]))
    penalizacao = np.sum(
        np.abs(superior[superior < 0]))+np.sum(np.abs(inferior[inferior > 0]))

    if relatorio == True:

        print('Penalização de Geração de Reativos:\n')
        print(penalizacao+penalizacaoext, '\n')
        print('Número de Violações:\n')
        print(len(inferior[inferior < 0])+len(superior[superior < 0]))

    else:

        return penalizacao+penalizacaoext

 ################################################################################################################################################################################################


def get_trafo_data(pes, relatorio=True):

    pes.trafo.sort_index()

    pes.res_trafo.sort_index()

    pes.trafo['tap_pos'] = pes.trafo['tap_pos']

    n_trafos_controlados = pes.trafo['tap_pos'].count()

    carregamento = pes.res_trafo['loading_percent'].to_numpy()/100

    tap_pos = pes.trafo[~pd.isnull(
        pes.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)

    tap_neutral = pes.trafo[~pd.isnull(
        pes.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)

    tap_step_percent = pes.trafo[~pd.isnull(
        pes.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)

    step = 0.01
    valores_taps = np.arange(start=0.9, stop=1.1, step=step)

    if relatorio == True:

        tap_pos = pes.trafo[~pd.isnull(
            pes.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)

        tap_neutral = pes.trafo[~pd.isnull(
            pes.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)

        tap_step_percent = pes.trafo[~pd.isnull(
            pes.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)

        valor_percentual = (tap_pos-tap_neutral)*(tap_step_percent/100) + 1

        if len(pes.bus) == 14:

            plt.figure(figsize=(20, 10))
            sns.scatterplot(x=np.arange(
                0, len(tap_pos)), y=valor_percentual, label='Valor do TAP', color='b', s=75)
            sns.lineplot(x=np.arange(0, len(tap_pos)), y=np.tile(
                [1.12], (len(tap_pos))), label='Limite Máximo do TAP', color='r')
            sns.lineplot(x=np.arange(0, len(tap_pos)), y=np.tile(
                [0.88], (len(tap_pos))), label='Limite Mínimo do TAP', color='orange')
            plt.grid()

        if len(pes.bus) == 30:

            plt.figure(figsize=(20, 10))
            sns.scatterplot(x=np.arange(
                0, len(tap_pos)), y=valor_percentual, label='Valor do TAP', color='b', s=75)
            sns.lineplot(x=np.arange(0, len(tap_pos)), y=np.tile(
                [1.12], (len(tap_pos))), label='Limite Máximo do TAP', color='r')
            sns.lineplot(x=np.arange(0, len(tap_pos)), y=np.tile(
                [0.88], (len(tap_pos))), label='Limite Mínimo do TAP', color='orange')
            plt.grid()

        if len(pes.bus) == 118:

            plt.figure(figsize=(20, 10))
            sns.scatterplot(x=np.arange(
                0, len(tap_pos)), y=valor_percentual, label='Valor do TAP', color='b', s=75)
            sns.lineplot(x=np.arange(0, len(tap_pos)), y=np.tile(
                [1.12], (len(tap_pos))), label='Limite Máximo do TAP', color='r')
            sns.lineplot(x=np.arange(0, len(tap_pos)), y=np.tile(
                [0.88], (len(tap_pos))), label='Limite Mínimo do TAP', color='orange')
            plt.grid()

        if len(pes.bus) == 300:

            plt.figure(figsize=(20, 10))
            sns.scatterplot(x=np.arange(
                0, len(tap_pos)), y=valor_percentual, label='Valor do TAP', color='b', s=75)
            sns.lineplot(x=np.arange(0, len(tap_pos)), y=np.tile(
                [1.12], (len(tap_pos))), label='Limite Máximo do TAP', color='r')
            sns.lineplot(x=np.arange(0, len(tap_pos)), y=np.tile(
                [0.88], (len(tap_pos))), label='Limite Mínimo do TAP', color='orange')
            plt.grid()

        if relatorio == True:
            plt.savefig("Limites de TAP.")

        print('Carregamento do Trafo (pu):\n')
        print(carregamento, '\n')
        print('Número de Trafos com TAP Controlado:\n')
        print(n_trafos_controlados, '\n')
        print('Valores dos TAPs:\n')
        print(valor_percentual, '\n')

    return tap_pos, tap_neutral, tap_step_percent, valores_taps

 ################################################################################################################################################################################################


def pen_trafo(linha, n_tap, n_vgen):
    '''    


    Valores dos TAPs Retirados de:

    - REFORMULAÇÃO DAS RESTRIÇÕESDE COMPLEMENTARIDADE EM PROBLEMAS DE FLUXO DE POTÊNCIA ÓTIMO
      Marina Valença Alencar - Dissertação de Mestrado

    - FUNÇÕES PENALIDADE PARA O TRATAMENTO DAS VARIÁVEIS DISCRETAS DO PROBLEMA DE FLUXO DE POTÊNCIA ÓTIMO REATIVO
      Daisy Paes Silva - Dissertação de Mestrado


    '''
    """    
   
    Calcula a penalização senoidal para taps não discretos.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    Parâmetros
    ----------   
    linha : linha da partícula.
    n_tap : número de taps.
    n_vgen : número de geradores.
    
    Retorno
    -------    
    linha : linha da partícula com os valores da penalização do trafo atualizados.
    
    
    Observações:
    ------------
    
    ...
    
    """

    step = 0.01

    linha[-3] = np.sum(np.square(np.sin((linha[n_vgen:n_vgen+n_tap]*np.pi/step))))

    return linha

 ################################################################################################################################################################################################


def get_bshunt_data(pes):
    '''    


    Valores dos Shunt Retirados de:

    - REFORMULAÇÃO DAS RESTRIÇÕESDE COMPLEMENTARIDADE EM PROBLEMAS DE FLUXO DE POTÊNCIA ÓTIMO
      Marina Valença Alencar - Dissertação de Mestrado

    - FUNÇÕES PENALIDADE PARA O TRATAMENTO DAS VARIÁVEIS DISCRETAS DO PROBLEMA DE FLUXO DE POTÊNCIA ÓTIMO REATIVO
      Daisy Paes Silva - Dissertação de Mestrado


    '''

    ieee118 = np.arange(0.00, 0.45, 0.001)
    ieee30 = np.arange(0.00, 0.35, 0.001)
    ieee300 = np.arange(start=-3.25, stop=3.25, step=0.1)

    bus = pes.shunt['bus'].sort_values().to_numpy()
    pes.shunt = pes.shunt.sort_index()

    if len(pes.bus) == 14:

        bsh = np.array([[0, 0.01, 0.02, 0.03, 0.04, 0.05]], dtype=object)

    if len(pes.bus) == 30:

        bsh = np.array([[0, 0.01, 0.02, 0.03, 0.04, 0.05], [
                       0, 0.01, 0.02, 0.03, 0.04, 0.05]], dtype=object)

    if len(pes.bus) == 118:

        bsh = np.array([np.arange(start=-0.40, stop=0, step=0.01),
                       np.arange(start=0, stop=0.14, step=0.01),
                       np.arange(start=-0.25, stop=0, step=0.01),
                       np.arange(start=0, stop=0.1, step=0.01),
                       np.arange(start=0, stop=0.1, step=0.01),
                       np.arange(start=0, stop=0.1, step=0.01),
                       np.arange(start=0, stop=0.15, step=0.01),
                       np.arange(start=0, stop=0.12, step=0.01),
                       np.arange(start=0, stop=0.2, step=0.01),
                       np.arange(start=0, stop=0.2, step=0.01),
                       np.arange(start=0, stop=0.1, step=0.01),
                       np.arange(start=0, stop=0.2, step=0.01),
                       np.arange(start=0, stop=0.06, step=0.01),
                       np.arange(start=0, stop=0.06, step=0.01)], dtype=object)

#         bsh = np.array([[-0.40, 0],
#                        [0, 0.06, 0.07, 0.13, 0.14, 0.20],
#                        [-0.25, 0],
#                        [0, 0.10],
#                        [0, 0.10],
#                         [0, 0.10],
#                         [0, 0.15],
#                         [0.08, 0.12, 0.20],
#                         [0, 0.10, 0.20],
#                         [0, 0.10, 0.20],
#                         [0, 0.10, 0.20],
#                         [0, 0.10, 0.20],
#                         [0, 0.06, 0.07, 0.13, 0.14, 0.20],
#                         [0, 0.06, 0.07, 0.13, 0.14, 0.20]],dtype=object)

    if len(pes.bus) == 300:

        bsh = np.array([[0, 1.5, 3, 4.5],  # 95
                        [0, 0.15, 0.30, 0.60],  # 98
                        [0, 0.15, 0.30, 0.45],  # 132
                        [-4.5, -3, -1.5, 0],  # 142
                        [-4.5, -3, -1.5, 0],  # 144
                        [0, 0.15, 0.30, 0.45, 0.60],  # 151
                        [0, 0.15, 0.30, 0.45, 0.60],  # 157
                        [-4.5, -3, -1.5, 0],  # 168
                        [-4.5, -3, -1.5, 0],  # 209
                        [-4.5, -3, -1.5, 0],  # 216
                        [-4.5, -3, -1.5, 0],  # 218
                        [0, 0.15, 0.30, 0.45, 0.60],  # 226
                        [0, 0.05, 0.01, 0.15],  # 267
                        [0],  # 274
                        [0],  # 276
                        [0],  # 277
                        [0],  # 278
                        [0],  # 279
                        [0],  # 280
                        [0],  # 281
                        [0, 0.05, 0.01, 0.15],  # 282
                        [0],  # 283
                        [0],  # 285
                        [0],  # 286
                        [0],  # 287
                        [0],  # 288
                        [0],  # 296
                        [0],  # 297
                        [0],  # 299
                        ], dtype=object)

#         bsh = np.array([[0,2,3.5,4.5], #95
#                 [0, 0.25, 0.44, 0.59], #98
#                 [0,0.19,0.34,0.39], #132
#                 [-4.5,0], #142
#                 [-4.5,0], #144
#                 [0, 0.25,0.44,0.59], #151
#                 [0, 0.25,0.44,0.59], #157
#                 [-2.5,0], #168
#                 [-4.5,0], #209
#                 [-4.5,0],#216
#                 [-1.5,0], #218
#                 [0, 0.25, 0.44, 0.59], #226
#                 [0, 0.15], #267
#                 [0], #274
#                 [0], #276
#                 [0], #277
#                 [0], #278
#                 [0], #279
#                 [0], #280
#                 [0], #281
#                 [0,0.15], #282
#                 [0], #283
#                 [0], #285
#                 [0], #286
#                 [0], #287
#                 [0], #288
#                 [0], #296
#                 [0], #297
#                 [0], #299
#                ],dtype=object)

    return bsh, bus

 ################################################################################################################################################################################################


def convert_trafo(tap_pos, tap_neutral, tap_step_percent, valores_taps):
    '''
    Converte TAPS conforme equação disponibilizada pelo pandapower.

    https://pandapower.readthedocs.io/en/v2.1.0/elements/trafo.html

    '''

    taps_convertido = tap_neutral + \
        ((valores_taps - 1.0)*(100/tap_step_percent))

    return taps_convertido

 ################################################################################################################################################################################################


def create_population(pes, num_individuals):
    vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior, barra = get_generator_data(
        pes, relatorio=False)

    n_vgen = len(vgen)+1

    vbus, theta, v_lim_superior, v_lim_inferior = get_vbus_data(
        pes, relatorio=True)

    tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
        pes, relatorio=True)

    n_taps = len(tap_pos)

    bshunt, bus = get_bshunt_data(pes)

    bshunt = np.array(bshunt)

    n_bshunt = len(bus)

    dimensao = n_taps + n_vgen + n_bshunt + 6

    enxame = np.zeros((num_individuals, dimensao), dtype=np.float64)

    enxame[:, 0:n_vgen] = np.random.uniform(
        np.max(v_lim_inferior), np.max(v_lim_superior), size=(num_individuals, n_vgen))

    enxame[:, n_vgen:n_vgen +
           n_taps] = np.random.choice(valores_taps, size=(num_individuals, n_taps))

    i = 1

    for bsh in bshunt:

        enxame[:, n_vgen+n_taps+i-1:n_vgen+n_taps +
               i] = np.random.choice(bsh, size=(num_individuals, 1))
        i = i+1

    return enxame


def create_population_v(pes, num_individuals):
    """"

    Cria o enxame de partículas.


    linhas = partículas

    colunas = tensões geradores, tap transformadores, susceptâncias shunt, perdas, penalização de tensão, penalização de reativo, penalização de trafo, penalização shunt, fitness

    """

    vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior, barra = get_generator_data(
        pes, relatorio=False)

    n_vgen = len(vgen)+1

    vbus, theta, v_lim_superior, v_lim_inferior = get_vbus_data(
        pes, relatorio=False)
    tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
        pes, relatorio=False)

    n_taps = len(tap_pos)

    bshunt, bus = get_bshunt_data(pes)

    bshunt = np.array(bshunt)

    n_bshunt = len(bus)

    dimensao = n_taps + n_vgen + n_bshunt + 6

    enxame = np.zeros((num_individuals, dimensao), dtype=np.float64)

    enxame[:, 0:n_vgen] = np.random.uniform(-1*np.ones(n_vgen)*np.max(
        v_lim_superior), np.ones(n_vgen)*np.max(v_lim_superior), size=(num_individuals, n_vgen))

    enxame[:, n_vgen:n_vgen+n_taps] = np.random.uniform(-1*np.ones(n_taps)*np.max(
        valores_taps), np.ones(n_taps)*np.max(valores_taps), size=(num_individuals, n_taps))

    i = 1

    for bsh in bshunt:

        enxame[:, n_vgen+n_taps+i-1:n_vgen+n_taps+i] = np.random.uniform(-1*np.ones(
            1)*np.max(bsh), np.ones(1)*np.max(bsh), size=(num_individuals, 1))
        i = i+1

    return enxame

 ################################################################################################################################################################################################


def pen_bshunt(grupo, n_tap, n_vgen, n_bshunt, pes):

    b = grupo[n_tap+n_vgen:n_tap+n_vgen+n_bshunt]

    bsh, bus = get_bshunt_data(pes)

    penal = 0

    i = 0

    bs = []

    for i in range(len(bsh)):

        bs.append(np.array(bsh[i]))

    for i in range(len(bs)):

        if len(bs[i][bs[i] <= b[i]]) == 0:
            penal = 1
            return penal
        if len(bs[i][bs[i] >= b[i]]) == 0:
            penal = 1
            return penal

        anterior = bs[i][bs[i] <= b[i]][-1]
        posterior = bs[i][bs[i] >= b[i]][0]
        alfa = np.pi*(np.ceil((anterior/(0.001+posterior-anterior))
                              )-(anterior/(0.001+posterior-anterior)))
        penal = penal + \
            np.square(np.sin((b[i]/(0.001+posterior-anterior))*np.pi+alfa))

    return penal

 ################################################################################################################################################################################################


def fluxo_de_pot(grupo, pes):

    n_bshunt = len(pes.shunt)
    n_vgen = len(pes.gen)+1
    n_tap = np.abs(pes.trafo['tap_pos']).count()

    matrizg, matrizz = conductance_matrix(pes, relatorio=False)

    for linha in range(grupo.shape[0]):
        pes.ext_grid['vm_pu'] = grupo[linha, 0]
        pes.gen['vm_pu'] = grupo[linha, 1:n_vgen]

        tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
            pes, relatorio=False)

        pes.trafo['tap_pos'][~pd.isnull(pes.trafo['tap_pos'])] = convert_trafo(
            tap_pos, tap_neutral, tap_step_percent, grupo[linha, n_vgen:n_vgen+n_tap])

        pes.shunt['q_mvar'] = grupo[linha, n_vgen +
                                    n_tap:n_vgen+n_tap+n_bshunt]*-100

        t1 = time.perf_counter()
        if len(pes.bus) == 300:
            pp.runpp(pes, algorithm='fdbx', numba=True, init='flat',
                     tolerance_mva=1e-5, max_iteration=100, trafo_model='pi')
        else:
            pp.runpp(pes, algorithm='fdbx', numba=True, init='results', tolerance_mva=1e-10,
                     max_iteration=100, enforce_q_lims=False, trafo_model='pi')
        t2 = time.perf_counter()

        vbus, theta, v_lim_superior, v_lim_inferior = get_vbus_data(
            pes, relatorio=False)

        grupo[linha, -6] = (pes.res_line['pl_mw'].sum()/100 + pes.res_trafo['pl_mw'].sum() /
                            100) + 0*np.sum((1-pes.res_bus['vm_pu'].values)**2)
#         grupo[linha,-6]=func_objetivo(vbus,theta,matrizg,matrizz,relatorio=False)

        grupo[linha, -5] = pen_tensao(vbus, v_lim_superior,
                                      v_lim_inferior, relatorio=False)

        vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior, barra = get_generator_data(
            pes, relatorio=False)

        grupo[linha, -4] = pen_ger_reativo(qgen, q_lim_superior,
                                           q_lim_inferior, pes, relatorio=False)

        grupo[linha, :] = pen_trafo(grupo[linha, :], n_tap, n_vgen)

        grupo[linha, -2] = pen_bshunt(grupo[linha, :],
                                      n_tap, n_vgen, n_bshunt, pes)

    return grupo


def fluxo_de_pot_q(grupo, pes):

    n_bshunt = len(pes.shunt)
    n_vgen = len(pes.gen)+1
    n_tap = np.abs(pes.trafo['tap_pos']).count()

    matrizg = conductance_matrix(pes, relatorio=False)

    for linha in range(grupo.shape[0]):

        pes.ext_grid['vm_pu'] = grupo[linha, 0]

        pes.gen['vm_pu'] = grupo[linha, 1:n_vgen]

        tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
            pes, relatorio=False)

        pes.trafo['tap_pos'][~pd.isnull(pes.trafo['tap_pos'])] = convert_trafo(
            tap_pos, tap_neutral, tap_step_percent, grupo[linha, n_vgen:n_vgen+n_tap])

        pes.shunt['q_mvar'] = grupo[linha, n_vgen +
                                    n_tap:n_vgen+n_tap+n_bshunt]*-100

        if len(pes.bus) == 300:

            pp.runpp(pes, algorithm='nr', numba=True, init='results',
                     tolerance_mva=1e-4, max_iteration=1000, enforce_q_lims=True)

        else:

            pp.runpp(pes, algorithm='nr', numba=True, init='results', tolerance_mva=1e-5,
                     max_iteration=1000, enforce_q_lims=True, trafo_model='pi')

        vbus, theta, v_lim_superior, v_lim_inferior = get_vbus_data(
            pes, relatorio=False)

        # func_objetivo(vbus,theta,matrizg,relatorio=False)
        grupo[linha, -6] = pes.res_line['pl_mw'].sum()/100 + \
            pes.res_trafo['pl_mw'].sum()/100

        grupo[linha, -5] = pen_tensao(vbus, v_lim_superior,
                                      v_lim_inferior, relatorio=False)

        vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior, barra = get_generator_data(
            pes, relatorio=False)

        grupo[linha, -4] = pen_ger_reativo(qgen, q_lim_superior,
                                           q_lim_inferior, pes, relatorio=False)

        grupo[linha, :] = pen_trafo(grupo[linha, :], n_tap, n_vgen)

        grupo[linha, -2] = pen_bshunt(grupo[linha, :],
                                      n_tap, n_vgen, n_bshunt, pes)

    return grupo

 ################################################################################################################################################################################################


def fitness(grupo, zeta, psi, sigma, omega):

    # fitness J       perdas         pen tensão         pen q mvar          pen trafo           pen bshunt
    grupo[:, -1] = (grupo[:, -6])+(zeta*grupo[:, -5]) + \
        (psi*grupo[:, -4])+(sigma*grupo[:, -3])+(omega*grupo[:, -2])

    return grupo

 ################################################################################################################################################################################################


def validacao(pes, best_solution, relatorio=True):

    valida = fluxo_de_pot(np.array([best_solution]), pes)

    if relatorio == True:
        print('Perdas de Potência Ativa [PU]:\n')
        print(valida[0][-6])
        print(' ')

        print('Penalização de Violação de Tensão [PU]:\n')
        print(valida[0][-5])
        print(' ')

        print('Penalização de Violação de Geração de Reativo [PU]:\n')
        print(valida[0][-4])
        print(' ')

        print('Penalização de Violação de TAP Discreto [PU]:\n')
        print(valida[0][-3])
        print(' ')

        print('Penalização de Violação de Bshunt Discreto [PU]:\n')
        print(valida[0][-2])
        print(' ')


def validacao_q(pes, best_solution, relatorio=True):

    valida = fluxo_de_pot_q(np.array([best_solution]), pes)

    if relatorio == True:
        print('Perdas de Potência Ativa [PU]:\n')
        print(valida[0][-6])
        print(' ')

        print('Penalização de Violação de Tensão [PU]:\n')
        print(valida[0][-5])
        print(' ')

        print('Penalização de Violação de Geração de Reativo [PU]:\n')
        print(valida[0][-4])
        print(' ')

        print('Penalização de Violação de TAP Discreto [PU]:\n')
        print(valida[0][-3])
        print(' ')

        print('Penalização de Violação de Bshunt Discreto [PU]:\n')
        print(valida[0][-2])
        print(' ')

 ################################################################################################################################################################################################


def discreto_bshunt(grupo, n_tap, n_vgen, n_bshunt, pes):

    b = grupo[n_tap+n_vgen:n_tap+n_vgen+n_bshunt]

    bsh, bus = get_bshunt_data(pes)

    penal = 0

    discretiza = []

    i = 0

    bs = []

    for i in range(len(bsh)):

        bs.append(np.array(bsh[i]))

    i = 0

    for c in bs:

        discretiza.append(c[np.argmin(np.abs(c-b[i]))])

        i = i+1

    return discretiza

 ################################################################################################################################################################################################


def discreto_tap(grupo, n_tap, n_vgen, n_bshunt, pes):

    b = grupo[n_vgen:n_vgen+n_tap]

    ref = np.arange(start=0.9, stop=1.1, step=0.01)

    discretizatap = np.zeros(len(b))

    i = 0

    for i in range(len(b)):

        discretizatap[i] = (ref[np.argmin(np.abs(ref-b[i]))])

    return discretizatap

 ################################################################################################################################################################################################


def otimizacao_pso_discreto(pes, zeta, psi, sigma, omega, max_iter, num_individuals, c1, c2, v_amp, valor_inicial, step, wmax, relatorio=True, inicial=True):

    enxame_fit = create_population(pes, n_particles)

    if len(pes.bus) == 14:

        n_vgen = 4+1
        n_tap = 3
        n_bshunt = 1

    if len(pes.bus) == 30:

        n_vgen = 5+1
        n_tap = 4
        n_bshunt = 2

    if len(pes.bus) == 118:

        n_vgen = 53+1
        n_tap = 9
        n_bshunt = 14

    if len(pes.bus) == 300:

        n_vgen = 68+1
        n_tap = 62
        n_bshunt = 29

    if inicial == True:

        enxame_fit[0, :] = valor_inicial

    w_max = wmax

    w_min = 0.4

    j = []

    tempo = []

    perdas = []

    pen_v = []

    pen_gq = []

    pen_tap = []

    pen_bsh = []

    v_lim_superior = np.repeat(pes.bus['max_vm_pu'][0], n_vgen)

    v_lim_inferior = np.repeat(pes.bus['min_vm_pu'][0], n_vgen)

    tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
        pes, relatorio=False)

    tap_max = np.repeat(valores_taps[-1], len(tap_pos))

    tap_min = np.repeat(valores_taps[0], len(tap_pos))

    bsh, b = get_bshunt_data(pes)

    bsh_max = []

    bsh_min = []

    for bs in bsh:
        bsh_max.append([np.max(bs)])
        bsh_min.append([np.min(bs)])

    maximo = np.expand_dims(np.concatenate(
        (v_lim_superior, tap_max, bsh_max), axis=None), 0)
    minimo = np.expand_dims(np.concatenate(
        (v_lim_inferior, tap_min, bsh_min), axis=None), 0)

    lim_sup = np.tile(maximo, (n_particles, 1))
    lim_inf = np.tile(minimo, (n_particles, 1))

    v_anterior = v_amp*create_population(pes, n_particles)

    xk = []

    for i in range(0, max_iter):

        start = time.time()

        r1 = np.random.random_sample(size=(n_particles, 1))

        r2 = np.random.random_sample(size=(n_particles, 1))

        enxame_fit_d = np.copy(enxame_fit)

        for linha in range(n_particles):

            enxame_fit_d[linha][n_vgen:n_vgen+n_tap] = discreto_tap(
                enxame_fit[linha].copy(), n_tap, n_vgen, n_bshunt, pes)
            enxame_fit_d[linha][n_vgen+n_tap:n_vgen+n_tap+n_bshunt] = discreto_bshunt(
                enxame_fit[linha].copy(), n_tap, n_vgen, n_bshunt, pes)

        enxame_fit[:, -6:] = (fluxo_de_pot(enxame_fit_d, pes))[:, -6:]

        enxame_fit[:, -6:] = (fitness(enxame_fit, zeta,
                              psi, sigma, omega))[:, -6:]

        xk.append(enxame_fit.copy())

        if i == 0:

            best_particles = enxame_fit.copy()

            global_best = best_particles[np.argsort(
                best_particles[:, -1])][0, :].copy()

            global_matriz = np.tile(global_best, (n_particles, 1))

            contador = 0

        for t in range(0, n_particles):

            if (enxame_fit[t, -1] < best_particles[t, -1]):

                best_particles[t, :] = enxame_fit[t, :].copy()

        global_best = best_particles[np.copy(
            np.argsort(best_particles[:, -1]))][0, :].copy()

        print(global_best)

        global_matriz = np.tile(global_best, (n_particles, 1))

        enxame_fit_anterior = enxame_fit.copy()

        w_novo = w_max-(w_max-w_min)*(i+1)/max_iter

        v_novo = np.multiply(w_novo, v_anterior.copy()) + c1*np.multiply(r1, (best_particles.copy(
        )-enxame_fit.copy())) + c2*np.multiply(r2, (global_matriz.copy()-enxame_fit.copy()))

        v_novo[:, :n_vgen+n_tap][v_novo[:, :n_vgen+n_tap] < -1*0.01] = -1*0.01

        v_novo[:, :n_vgen+n_tap][v_novo[:, :n_vgen+n_tap] > 1*0.01] = 1*0.01

        v_novo[:, n_vgen+n_tap:n_vgen+n_tap+n_bshunt][v_novo[:,
                                                             n_vgen+n_tap:n_vgen+n_tap+n_bshunt] > 1*0.01] = 1*0.01

        v_novo[:, n_vgen+n_tap:n_vgen+n_tap+n_bshunt][v_novo[:,
                                                             n_vgen+n_tap:n_vgen+n_tap+n_bshunt] < -1*0.01] = -1*0.01


#         if i<1:
#             enxame_fit_novo = enxame_fit_anterior  + v_novo

#         if i>=1:


#             for p in range(0,n_particles):


#                 for coluna in range(len(global_best)):

#                     if xk[i][p,-1] > xk[i-1][p,-1]:

#                         enxame_fit_novo[p,coluna] = xk[i][p,coluna].copy()  + v_novo[p,coluna]

#                     if xk[i][p,-1] < xk[i-1][p,-1]:

#                         if xk[i][p,coluna] > xk[i-1][p,coluna]:

#                             enxame_fit_novo[p,coluna] = xk[i][p,coluna].copy()  + 1*np.abs(v_novo[p,coluna])

#                         if xk[i][p,coluna] == xk[i-1][p,coluna]:

#                             enxame_fit_novo[p,coluna] = xk[i][p,coluna].copy()  + 0*v_novo[p,coluna]

#                         if xk[i][p,coluna] < xk[i-1][p,coluna]:

#                             enxame_fit_novo[p,coluna] = xk[i][p,coluna].copy() + -1*np.abs(v_novo[p,coluna])

        enxame_fit_novo = enxame_fit_anterior + v_novo

        v_anterior = v_novo.copy()

#         delta = lim_sup[0,:]-lim_inf[0,:]

#         step = 0.25

#         for k in range(int(n_particles/10)):

#             mutacao=(np.random.randn(len(global_best[:-6])))*step*delta

#             posi = np.random.randint(0,n_particles)

#             enxame_fit_novo[posi,:-6] = enxame_fit_novo[posi,:-6] + mutacao


#         for linha in range(n_particles):

#             enxame_fit_novo[linha][n_vgen:n_vgen+n_tap] = discreto_tap(enxame_fit_novo[linha],n_tap,n_vgen,n_bshunt,pes)
#             enxame_fit_novo[linha][n_vgen+n_tap:n_vgen+n_tap+n_bshunt] = discreto_bshunt(enxame_fit_novo[linha],n_tap,n_vgen,n_bshunt,pes)

        enxame_estat = enxame_fit_novo[:, -6:]

        enxame_fit = np.concatenate((np.clip(
            enxame_fit_novo[:, 0:-6], a_min=lim_inf, a_max=lim_sup, out=enxame_fit_novo[:, 0:-6]), enxame_estat), axis=1)

        end = time.time()

        elapsed = end - start

        j.append(global_best[-1])

        perdas.append(global_best[-6])

        pen_v.append(global_best[-5])

        pen_gq.append(global_best[-4])

        pen_tap.append(global_best[-3])

        pen_bsh.append(global_best[-2])

        tempo.append(elapsed)

        if relatorio == True:

            print(' ')

            print('Melhor Global da Iteração:', i)

            print('Perdas (pu):', global_best[-6])

            print('Penalização de Tensão:', global_best[-5])

            print('Penalização de Geração de Reativo:', global_best[-4])

            print('Fitness:', global_best[-1])

            print('Tempo: ', elapsed)

            print(' ')

            print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')

    if relatorio == True:

        plt.figure(figsize=(18, 10))
        plt.plot(perdas)
        plt.savefig("Perdas.png")

        plt.title('Otimização por AG')
        plt.ylabel('Perdas de Potência Ativa (pu)')
        plt.xlabel('Número da Iteração')

        plt.figure(figsize=(18, 10))
        plt.plot(j)
        plt.savefig("Otimizacao.png")

        plt.title('Otimização por AG')
        plt.ylabel('Fitness (J)')
        plt.xlabel('Número da Iteração')

        plt.figure(figsize=(18, 10))
        plt.plot(pen_v)
        plt.savefig("Otimizacao2.png")

        plt.title('Otimização por AG')
        plt.ylabel('Penalização de Tensão')
        plt.xlabel('Número da Iteração')

        plt.figure(figsize=(18, 10))
        plt.plot(pen_gq)
        plt.title('Otimização por AG')
        plt.ylabel('Penalização de Geração de Reativo')
        plt.xlabel('Número da Iteração')

        plt.figure(figsize=(18, 10))
        plt.plot(pen_tap)
        plt.title('Otimização por AG')
        plt.ylabel('Penalização do TAP')
        plt.xlabel('Número da Iteração')

        plt.figure(figsize=(18, 10))
        plt.plot(pen_bsh)
        plt.title('Otimização por AG')
        plt.ylabel('Penalização do BShunt')
        plt.xlabel('Número da Iteração')

    return j, perdas, pen_v, pen_gq, pen_tap, pen_bsh, global_best, tempo

 ################################################################################################################################################################################################


def optimize_ga(pes, pt, rgap, zeta, psi, sigma, omega, max_iter, num_individuals, c1, c2, v_amp, valor_inicial, step, wmax, relatorio=True, inicial=True):
    individuals_fit = create_population(pes, num_individuals)

    if len(pes.bus) == 14:

        n_vgen = 4+1
        n_tap = 3
        n_bshunt = 1

    if len(pes.bus) == 30:

        n_vgen = 5+1
        n_tap = 4
        n_bshunt = 2

    if len(pes.bus) == 118:

        n_vgen = 53+1
        n_tap = 9
        n_bshunt = 14

    if len(pes.bus) == 300:

        n_vgen = 68+1
        n_tap = 62
        n_bshunt = 29

    if inicial == True:
        individuals_fit[0, :] = valor_inicial

    w_max = wmax

    w_min = 0.4

    j = []

    tempo = []

    perdas = []

    pen_v = []

    pen_gq = []

    pen_tap = []

    pen_bsh = []

    refresh_rate = np.zeros((num_individuals))

    v_lim_superior = np.repeat(pes.bus['max_vm_pu'][0], n_vgen)

    v_lim_inferior = np.repeat(pes.bus['min_vm_pu'][0], n_vgen)

    tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
        pes, relatorio=False)

    tap_max = np.repeat(valores_taps[-1], len(tap_pos))

    tap_min = np.repeat(valores_taps[0], len(tap_pos))

    bsh, b = get_bshunt_data(pes)

    bsh_max = []

    bsh_min = []

    for bs in bsh:
        bsh_max.append([np.max(bs)])
        bsh_min.append([np.min(bs)])

    maximo = np.expand_dims(np.concatenate(
        (v_lim_superior, tap_max, bsh_max), axis=None), 0)
    minimo = np.expand_dims(np.concatenate(
        (v_lim_inferior, tap_min, bsh_min), axis=None), 0)

    lim_sup = np.tile(maximo, (num_individuals, 1))
    lim_inf = np.tile(minimo, (num_individuals, 1))

    v_anterior = v_amp * create_population(pes, num_individuals)

    xk = []

    for i in range(0, max_iter):

        start = time.time()

        r1 = np.random.random_sample(size=(num_individuals, 1))

        r2 = np.random.random_sample(size=(num_individuals, 1))

        individuals_fit_d = np.copy(individuals_fit)

        for linha in range(num_individuals):
            individuals_fit_d[linha][n_vgen:n_vgen+n_tap] = discreto_tap(
                individuals_fit[linha].copy(), n_tap, n_vgen, n_bshunt, pes)
            individuals_fit_d[linha][n_vgen+n_tap:n_vgen+n_tap+n_bshunt] = discreto_bshunt(
                individuals_fit[linha].copy(), n_tap, n_vgen, n_bshunt, pes)

        individuals_fit[:, -
                        6:] = (fluxo_de_pot(individuals_fit_d, pes))[:, -6:]

        individuals_fit[:, -6:] = (fitness(individuals_fit, zeta,
                                           psi, sigma, omega))[:, -6:]

        xk.append(individuals_fit.copy())

        t1 = time.perf_counter()
        if i == 0:

            best_particles = individuals_fit.copy()

            global_best = best_particles[np.argsort(
                best_particles[:, -1])][0, :].copy()

            global_matriz = np.tile(global_best, (num_individuals, 1))

            contador = 0

            pbestfi = np.copy(best_particles)

        for t in range(0, num_individuals):

            if (individuals_fit[t, -1] < best_particles[t, -1]):

                best_particles[t, :] = individuals_fit[t, :].copy()

        global_best = best_particles[np.copy(
            np.argsort(best_particles[:, -1]))][0, :].copy()

        global_best[n_vgen:n_vgen+n_tap] = discreto_tap(
            global_best.copy(), n_tap, n_vgen, n_bshunt, pes)

        global_best[n_vgen+n_tap:n_vgen+n_tap+n_bshunt] = discreto_bshunt(
            global_best.copy(), n_tap, n_vgen, n_bshunt, pes)

        for li in range(num_individuals):

            if refresh_rate[li] >= rgap:

                pbestfi[li, :] = best_particles[li, :]

                refresh_rate[li] = 0

                for co in range(len(global_best)-7):

                    if np.random.rand() <= pt:

                        p1 = int(np.floor(num_individuals*np.random.rand()))

                        p2 = int(np.floor(num_individuals*np.random.rand()))

                        while (p1 == p2):

                            p1 = int(
                                np.floor(num_individuals*np.random.rand()))

                            p2 = int(
                                np.floor(num_individuals*np.random.rand()))

                        part1 = best_particles[p1, :]

                        part2 = best_particles[p2, :]

                        if part1[-1] <= part2[-1]:

                            pbestfi[li, co] = part1[co]

                        else:
                            pbestfi[li, co] = part2[co]

        global_matriz = np.tile(global_best, (num_individuals, 1))

        individuals_fit_anterior = individuals_fit.copy()

        w_novo = w_max-(w_max-w_min)*(i+1)/max_iter

        v_novo = np.multiply(w_novo, v_anterior.copy()) + \
            c1*np.multiply(r1, (pbestfi-individuals_fit.copy()))

        v_novo[:, :n_vgen+n_tap][v_novo[:, :n_vgen+n_tap] < -1*step] = -1*step

        v_novo[:, :n_vgen+n_tap][v_novo[:, :n_vgen+n_tap] > 1*step] = 1*step

        v_novo[:, n_vgen+n_tap:n_vgen+n_tap+n_bshunt][v_novo[:,
                                                             n_vgen+n_tap:n_vgen+n_tap+n_bshunt] > 1*step] = 1*step

        v_novo[:, n_vgen+n_tap:n_vgen+n_tap+n_bshunt][v_novo[:,
                                                             n_vgen+n_tap:n_vgen+n_tap+n_bshunt] < -1*step] = -1*step

        individuals_fit_novo = individuals_fit_anterior + v_novo

        v_anterior = v_novo.copy()

        individuals_estat = individuals_fit_novo[:, -6:]

        individuals_fit = np.concatenate((np.clip(
            individuals_fit_novo[:, 0:-6], a_min=lim_inf, a_max=lim_sup, out=individuals_fit_novo[:, 0:-6]), individuals_estat), axis=1)

        end = time.time()

        elapsed = end - start

        j.append(global_best[-1])

        perdas.append(global_best[-6])

        pen_v.append(global_best[-5])

        pen_gq.append(global_best[-4])

        pen_tap.append(global_best[-3])

        pen_bsh.append(global_best[-2])

        tempo.append(elapsed)

        if relatorio == True:

            print(' ')

            print('Melhor Global da Iteração:', i)

            print('Perdas (pu):', global_best[-6])

            print('Penalização de Tensão:', global_best[-5])

            print('Penalização de Geração de Reativo:', global_best[-4])

            print('Fitness:', global_best[-1])

            print('Tempo: ', elapsed)

            print(' ')

            print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')

    if relatorio == True:

        plt.figure(figsize=(18, 10))
        plt.plot(perdas)

        plt.title('Otimização por AG - Perdas de Potência Ativa')
        plt.ylabel('Perdas de Potência Ativa (pu)')
        plt.xlabel('Número da Iteração')
        plt.savefig("Otimizacao1.png")

        plt.figure(figsize=(18, 10))
        plt.plot(j)

        plt.title('Otimização por AG - Função Custo')
        plt.ylabel('Fitness (J)')
        plt.xlabel('Número da Iteração')
        plt.savefig("Otimizacao2.png")

        plt.figure(figsize=(18, 10))
        plt.plot(pen_v)
        plt.title('Otimização por AG - Penalização de Tensão')
        plt.ylabel('Penalização de Tensão')
        plt.xlabel('Número da Iteração')
        plt.savefig("Otimizacao3.png")

        plt.figure(figsize=(18, 10))
        plt.plot(pen_gq)
        plt.title('Otimização por AG - Penalização de Geração de Reativo')
        plt.ylabel('Penalização de Geração de Reativo')
        plt.xlabel('Número da Iteração')
        plt.savefig("Otimizacao4.png")

        plt.figure(figsize=(18, 10))
        plt.plot(pen_tap)
        plt.title('Otimização por AG - Penalização do TAP')
        plt.ylabel('Penalização do TAP')
        plt.xlabel('Número da Iteração')
        plt.savefig("Otimizacao5.png")

        plt.figure(figsize=(18, 10))
        plt.plot(pen_bsh)
        plt.title('Otimização por AG - Penalização do BShunt')
        plt.ylabel('Penalização do BShunt')
        plt.xlabel('Número da Iteração')
        plt.savefig("Otimizacao6.png")

    return j, perdas, pen_v, pen_gq, pen_tap, pen_bsh, global_best, tempo

 ################################################################################################################################################################################################


def create_population_fpo(pes, n_particulas):
    """"

    Cria o enxame de partículas.


    linhas = partículas

    colunas = tensões geradores, tap transformadores, susceptâncias shunt, perdas, penalização de tensão, penalização de reativo, penalização de trafo, penalização shunt, fitness

    """

    vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior, barra = get_generator_data(
        pes, relatorio=False)

    n_vgen = len(vgen)+1

    n_pot = len(vgen)

    vbus, theta, v_lim_superior, v_lim_inferior = get_vbus_data(
        pes, relatorio=False)

    tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
        pes, relatorio=False)

    n_taps = len(tap_pos)

    bshunt, bus = get_bshunt_data(pes)

    bshunt = np.array(bshunt)

    n_bshunt = len(bus)

    dimensao = n_taps + n_vgen + n_bshunt + n_pot + 7

    enxame = np.zeros((n_particulas, dimensao), dtype=np.float64)

    max_pot = pes.gen['max_p_mw'].values

    min_pot = pes.gen['min_p_mw'].values

    enxame[:, 0:n_vgen] = np.random.uniform(
        np.max(v_lim_inferior), np.max(v_lim_superior), size=(n_particulas, n_vgen))

    enxame[:, n_vgen:n_vgen +
           n_taps] = np.random.choice(valores_taps, size=(n_particulas, n_taps))

    i = 1

    for bsh in bshunt:

        enxame[:, n_vgen+n_taps+i-1:n_vgen+n_taps +
               i] = np.random.choice(bsh, size=(n_particulas, 1))
        i = i+1

    for i in range(n_pot):

        a = np.random.uniform(max_pot[i], min_pot[i], size=(n_particulas))

        enxame[:, n_vgen+n_taps+n_bshunt+i] = a/100

    return enxame


def fluxo_de_pot_fpo(grupo, pes):

    n_bshunt = len(pes.shunt)

    n_vgen = len(pes.gen)+1

    n_tap = np.abs(pes.trafo['tap_pos']).count()

    n_pot = len(pes.gen)


#         custo

    polycosts = pes.poly_cost[pes.poly_cost['et'] != 'ext_grid']

    index_list = pes.gen.index.values.tolist()

    a_k = []
    b_k = []
    c_k = []

    for val in index_list:

        a_k.append(polycosts['cp2_eur_per_mw2']
                   [polycosts['element'] == val].values[0])
        b_k.append(polycosts['cp1_eur_per_mw']
                   [polycosts['element'] == val].values[0])
        c_k.append(polycosts['cp0_eur'][polycosts['element'] == val].values[0])

    pmax0 = pes.ext_grid['max_p_mw'].values
    pmin0 = pes.ext_grid['min_p_mw'].values

    a_k0 = pes.poly_cost['cp2_eur_per_mw2'][pes.poly_cost['et']
                                            == 'ext_grid'].values

    b_k0 = pes.poly_cost['cp1_eur_per_mw'][pes.poly_cost['et']
                                           == 'ext_grid'].values

    c_k0 = pes.poly_cost['cp0_eur'][pes.poly_cost['et'] == 'ext_grid'].values

    e_k0 = (5/100)*((a_k0*pmax0**2 + b_k0*pmax0 +
                     c_k0 + a_k0*pmin0**2 + b_k0*pmin0 + c_k0)/2)

    f_k0 = (
        4*np.pi/(pes.ext_grid['max_p_mw'].values-pes.ext_grid['min_p_mw'].values))

    e_k = (5/100)*((a_k*pes.gen['max_p_mw'].values**2 + b_k*pes.gen['max_p_mw'].values +
                    c_k + a_k*pes.gen['min_p_mw'].values**2 + b_k*pes.gen['min_p_mw'].values + c_k)/2)

    f_k = (4*np.pi/(pes.gen['max_p_mw'].values-pes.gen['min_p_mw'].values))

    a_k_t = np.concatenate((a_k0, np.array(a_k)))

    b_k_t = np.concatenate((b_k0, np.array(b_k)))

    c_k_t = np.concatenate((c_k0, np.array(c_k)))

    e_k_t = np.concatenate((e_k0, np.array(e_k)))

    f_k_t = np.concatenate((f_k0, np.array(f_k)))

    pg_min0 = pes.ext_grid['min_p_mw'].values

    pg_mins = pes.gen['min_p_mw'].values

    pg_min = np.concatenate((pg_min0, pg_mins))

    pg0 = pes.res_ext_grid['p_mw'].values

    for linha in range(grupo.shape[0]):

        pes.ext_grid['vm_pu'] = grupo[linha, 0]

        pes.gen['vm_pu'] = grupo[linha, 1:n_vgen]

        tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
            pes, relatorio=False)

        pes.trafo['tap_pos'][~pd.isnull(pes.trafo['tap_pos'])] = convert_trafo(
            tap_pos, tap_neutral, tap_step_percent, grupo[linha, n_vgen:n_vgen+n_tap])

        pes.shunt['q_mvar'] = grupo[linha, n_vgen +
                                    n_tap:n_vgen+n_tap+n_bshunt]*-100

        pes.gen['p_mw'] = grupo[linha, n_vgen+n_tap +
                                n_bshunt:n_vgen+n_tap+n_bshunt+n_pot]*100

        if len(pes.bus) == 300:

            pp.runpp(pes, algorithm='fdbx', numba=True, init='flat',
                     tolerance_mva=1e-4, max_iteration=100000, trafo_model='pi')

        else:

            pp.runpp(pes, algorithm='fdbx', numba=True, init='results', tolerance_mva=1e-7,
                     max_iteration=1000, enforce_q_lims=False, trafo_model='pi')

#             pp.rundcpp(pes)

        vbus, theta, v_lim_superior, v_lim_inferior = get_vbus_data(
            pes, relatorio=False)

# #         # perdas

#         grupo[linha,-7] = (pes.res_line['pl_mw'].sum()/100 + pes.res_trafo['pl_mw'].sum()/100)

# #         desvio de tensao

#         grupo[linha,-7] = np.sum(np.abs(1-pes.res_bus['vm_pu'].values))

        pgs = grupo[linha, n_vgen+n_tap +
                    n_bshunt:n_vgen+n_tap+n_bshunt+n_pot]*100

        pg = np.concatenate((pg0, pgs))

        c = np.abs(np.sum((pg**2)*a_k_t+pg*b_k_t+c_k_t +
                   np.abs(e_k_t*np.sin(f_k_t*(pg_min-pg)))))

        grupo[linha, -7] = c

        grupo[linha, -6] = pen_tensao(vbus, v_lim_superior,
                                      v_lim_inferior, relatorio=False)

        vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior, barra = get_generator_data(
            pes, relatorio=False)

        grupo[linha, -5] = pen_ger_reativo(qgen, q_lim_superior,
                                           q_lim_inferior, pes, relatorio=False)

        grupo[linha, :] = pen_trafo(grupo[linha, :], n_tap, n_vgen)

        grupo[linha, -3] = pen_bshunt(grupo[linha, :],
                                      n_tap, n_vgen, n_bshunt, pes)

        # pen gerador 0

        pg_0 = pes.res_ext_grid['p_mw'].values

        pg_0_min = pes.ext_grid['min_p_mw'].values

        pg_0_max = pes.ext_grid['max_p_mw'].values

        pen = 0

        if pg_0 > pg_0_max:

            pen = np.abs((pg_0_max-pg_0))

        elif pg_0 < pg_0_min:

            pen = np.abs((pg_0-pg_0_min))

        else:
            pass

        if len(pes.bus) == 30:

            if 55 <= pg_0 <= 66:

                vector = np.array([55, 66])
                posi = np.argmin(np.abs(np.array([55, 66])-pg_0))
                pen = pen + np.abs(vector[posi]-pg_0)

            elif 80 <= pg_0 <= 120:

                vector = np.array([80, 120])
                posi = np.argmin(np.abs(np.array([80, 120])-pg_0))
                pen = pen + np.abs(vector[posi]-pg_0)

            else:
                pass

        if len(pes.bus) == 14:
            if (1/7)*332.4 <= pg_0 <= (2/7)*332.4:

                mean = (((1/7)*332.4)+((2/7)*332.4))/2

                if pg_0 <= mean:

                    pen = pen + np.abs((1/7)*332.4-pg_0)

                else:

                    pen = pen + np.abs((2/7)*332.4-pg_0)

            elif (3/7)*332.4 <= pg_0 <= (4/7)*332.4:

                mean = (((3/7)*332.4)+((4/7)*332.4))/2

                if pg_0 <= mean:

                    pen = pen + np.abs((3/7)*332.4-pg_0)

                else:

                    pen = pen + np.abs((4/7)*332.4-pg_0)

            elif (5/7)*332.4 <= pg_0 <= (6/7)*332.4:

                mean = (((5/7)*332.4)+((6/7)*332.4))/2

                if pg_0 <= mean:

                    pen = pen + np.abs((5/7)*332.4-pg_0)

                else:

                    pen = pen + np.abs((6/7)*332.4-pg_0)

        grupo[linha, -2] = pen

    return grupo


def fluxo_de_pot_fpo_dc(grupo, pes):

    n_bshunt = len(pes.shunt)
    n_vgen = len(pes.gen)+1
    n_tap = np.abs(pes.trafo['tap_pos']).count()
    n_pot = len(pes.gen)


#         custo

    polycosts = pes.poly_cost[pes.poly_cost['et'] != 'ext_grid']

    index_list = pes.gen.index.values.tolist()

    a_k = []
    b_k = []
    c_k = []

    for val in index_list:

        a_k.append(polycosts['cp2_eur_per_mw2']
                   [polycosts['element'] == val].values[0])
        b_k.append(polycosts['cp1_eur_per_mw']
                   [polycosts['element'] == val].values[0])
        c_k.append(polycosts['cp0_eur'][polycosts['element'] == val].values[0])

    pmax0 = pes.ext_grid['max_p_mw'].values
    pmin0 = pes.ext_grid['min_p_mw'].values

    a_k0 = pes.poly_cost['cp2_eur_per_mw2'][pes.poly_cost['et']
                                            == 'ext_grid'].values
    b_k0 = pes.poly_cost['cp1_eur_per_mw'][pes.poly_cost['et']
                                           == 'ext_grid'].values
    c_k0 = pes.poly_cost['cp0_eur'][pes.poly_cost['et'] == 'ext_grid'].values
    e_k0 = (5/100)*((a_k0*pmax0**2 + b_k0*pmax0 +
                     c_k0 + a_k0*pmin0**2 + b_k0*pmin0 + c_k0)/2)
    f_k0 = (
        4*np.pi/(pes.ext_grid['max_p_mw'].values-pes.ext_grid['min_p_mw'].values))

    e_k = (5/100)*((a_k*pes.gen['max_p_mw'].values**2 + b_k*pes.gen['max_p_mw'].values +
                    c_k + a_k*pes.gen['min_p_mw'].values**2 + b_k*pes.gen['min_p_mw'].values + c_k)/2)

    f_k = (4*np.pi/(pes.gen['max_p_mw'].values-pes.gen['min_p_mw'].values))

    a_k_t = np.concatenate((a_k0, np.array(a_k)))
    b_k_t = np.concatenate((b_k0, np.array(b_k)))
    c_k_t = np.concatenate((c_k0, np.array(c_k)))
    e_k_t = np.concatenate((e_k0, np.array(e_k)))
    f_k_t = np.concatenate((f_k0, np.array(f_k)))

    pg_min0 = pes.ext_grid['min_p_mw'].values

    pg_mins = pes.gen['min_p_mw'].values

    pg_min = np.concatenate((pg_min0, pg_mins))

    pg0 = pes.res_ext_grid['p_mw'].values

    matrizg = conductance_matrix(pes, relatorio=False)

    for linha in range(grupo.shape[0]):

        pes.ext_grid['vm_pu'] = grupo[linha, 0]

        pes.gen['vm_pu'] = grupo[linha, 1:n_vgen]

        tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
            pes, relatorio=False)

        pes.trafo['tap_pos'][~pd.isnull(pes.trafo['tap_pos'])] = convert_trafo(
            tap_pos, tap_neutral, tap_step_percent, grupo[linha, n_vgen:n_vgen+n_tap])

        pes.shunt['q_mvar'] = grupo[linha, n_vgen +
                                    n_tap:n_vgen+n_tap+n_bshunt]*-100

        pes.gen['p_mw'] = grupo[linha, n_vgen+n_tap +
                                n_bshunt:n_vgen+n_tap+n_bshunt+n_pot]*100

        if len(pes.bus) == 300:

            pp.runpp(pes, algorithm='fdbx', numba=True, init='flat',
                     tolerance_mva=1e-4, max_iteration=100000, trafo_model='pi')

        else:

            #             pp.runpp(pes,algorithm='nr',numba=True, init = 'results', tolerance_mva = 1e-8,max_iteration=1000,enforce_q_lims=False,trafo_model='pi')

            pp.rundcpp(pes)

        vbus, theta, v_lim_superior, v_lim_inferior = get_vbus_data(
            pes, relatorio=False)

# #         # perdas

#         grupo[linha,-7] = (pes.res_line['pl_mw'].sum()/100 + pes.res_trafo['pl_mw'].sum()/100)

# #         desvio de tensao

#         grupo[linha,-7] = np.sum(np.abs(1-pes.res_bus['vm_pu'].values))

        pgs = grupo[linha, n_vgen+n_tap +
                    n_bshunt:n_vgen+n_tap+n_bshunt+n_pot]*100

        pg = np.concatenate((pg0, pgs))

        c = np.abs(np.sum((pg**2)*a_k_t+pg*b_k_t+c_k_t +
                   np.abs(e_k_t*np.sin(f_k_t*(pg_min-pg)))))

        grupo[linha, -7] = c

        grupo[linha, -6] = pen_tensao(vbus, v_lim_superior,
                                      v_lim_inferior, relatorio=False)

        vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior, barra = get_generator_data(
            pes, relatorio=False)

        grupo[linha, -5] = pen_ger_reativo(qgen, q_lim_superior,
                                           q_lim_inferior, pes, relatorio=False)

        grupo[linha, :] = pen_trafo(grupo[linha, :], n_tap, n_vgen)

        grupo[linha, -3] = pen_bshunt(grupo[linha, :],
                                      n_tap, n_vgen, n_bshunt, pes)

        # pen gerador 0

        pg_0 = pes.res_ext_grid['p_mw'].values

        pg_0_min = pes.ext_grid['min_p_mw'].values

        pg_0_max = pes.ext_grid['max_p_mw'].values

        pen = 0

        if pg_0 > pg_0_max:

            pen = np.abs((pg_0_max-pg_0))

        elif pg_0 < pg_0_min:

            pen = np.abs((pg_0-pg_0_min))

        else:
            pass

        if len(pes.bus) == 30:

            if 55 <= pg_0 <= 66:

                vector = np.array([55, 66])
                posi = np.argmin(np.abs(np.array([55, 66])-pg_0))
                pen = pen + np.abs(vector[posi]-pg_0)

            elif 80 <= pg_0 <= 120:

                vector = np.array([80, 120])
                posi = np.argmin(np.abs(np.array([80, 120])-pg_0))
                pen = pen + np.abs(vector[posi]-pg_0)

            else:
                pass

        if len(pes.bus) == 14:
            if (1/7)*332.4 <= pg_0 <= (2/7)*332.4:

                mean = (((1/7)*332.4)+((2/7)*332.4))/2

                if pg_0 <= mean:

                    pen = pen + np.abs((1/7)*332.4-pg_0)

                else:

                    pen = pen + np.abs((2/7)*332.4-pg_0)

            elif (3/7)*332.4 <= pg_0 <= (4/7)*332.4:

                mean = (((3/7)*332.4)+((4/7)*332.4))/2

                if pg_0 <= mean:

                    pen = pen + np.abs((3/7)*332.4-pg_0)

                else:

                    pen = pen + np.abs((4/7)*332.4-pg_0)

            elif (5/7)*332.4 <= pg_0 <= (6/7)*332.4:

                mean = (((5/7)*332.4)+((6/7)*332.4))/2

                if pg_0 <= mean:

                    pen = pen + np.abs((5/7)*332.4-pg_0)

                else:

                    pen = pen + np.abs((6/7)*332.4-pg_0)

        grupo[linha, -2] = pen

    return grupo


def fluxo_de_pot_fpo_q(grupo, pes):

    n_bshunt = len(pes.shunt)
    n_vgen = len(pes.gen)+1
    n_tap = np.abs(pes.trafo['tap_pos']).count()
    n_pot = len(pes.gen)

    matrizg = conductance_matrix(pes, relatorio=False)

    for linha in range(grupo.shape[0]):

        pes.ext_grid['vm_pu'] = grupo[linha, 0]

        pes.gen['vm_pu'] = grupo[linha, 1:n_vgen]

        tap_pos, tap_neutral, tap_step_percent, valores_taps = get_trafo_data(
            pes, relatorio=False)

        pes.trafo['tap_pos'][~pd.isnull(pes.trafo['tap_pos'])] = convert_trafo(
            tap_pos, tap_neutral, tap_step_percent, grupo[linha, n_vgen:n_vgen+n_tap])

        pes.shunt['q_mvar'] = grupo[linha, n_vgen +
                                    n_tap:n_vgen+n_tap+n_bshunt]*-100

        pes.gen['p_mw'] = grupo[linha, n_vgen+n_tap +
                                n_bshunt:n_vgen+n_tap+n_bshunt+n_pot]*100

        if len(pes.bus) == 300:

            pp.runpp(pes, algorithm='fdbx', numba=True, init='flat',
                     tolerance_mva=1e-6, max_iteration=10000, trafo_model='pi')

        else:

            pp.runpp(pes, algorithm='nr', numba=True, init='results', tolerance_mva=1e-6,
                     max_iteration=1000, enforce_q_lims=True, trafo_model='pi')

        vbus, theta, v_lim_superior, v_lim_inferior = get_vbus_data(
            pes, relatorio=False)

#         # perdas

#         grupo[linha,-7] = (pes.res_line['pl_mw'].sum()/100 + pes.res_trafo['pl_mw'].sum()/100)

#         desvio de tensao

#         grupo[linha,-7] = np.sum(np.abs(1-pes.res_bus['vm_pu'].values))

#         custo
#         custo

        polycosts = pes.poly_cost[pes.poly_cost['et'] != 'ext_grid']

        index_list = pes.gen.index.values.tolist()

        a_k = []
        b_k = []
        c_k = []

        for val in index_list:

            a_k.append(polycosts['cp2_eur_per_mw2']
                       [polycosts['element'] == val].values[0])
            b_k.append(polycosts['cp1_eur_per_mw']
                       [polycosts['element'] == val].values[0])
            c_k.append(polycosts['cp0_eur']
                       [polycosts['element'] == val].values[0])

        pmax0 = pes.ext_grid['max_p_mw'].values
        pmin0 = pes.ext_grid['min_p_mw'].values

        a_k0 = pes.poly_cost['cp2_eur_per_mw2'][pes.poly_cost['et']
                                                == 'ext_grid'].values
        b_k0 = pes.poly_cost['cp1_eur_per_mw'][pes.poly_cost['et']
                                               == 'ext_grid'].values
        c_k0 = pes.poly_cost['cp0_eur'][pes.poly_cost['et']
                                        == 'ext_grid'].values
        e_k0 = (5/100)*((a_k0*pmax0**2 + b_k0*pmax0 +
                         c_k0 + a_k0*pmin0**2 + b_k0*pmin0 + c_k0)/2)
        f_k0 = (
            4*np.pi/(pes.ext_grid['max_p_mw'].values-pes.ext_grid['min_p_mw'].values))

        e_k = (5/100)*((a_k*pes.gen['max_p_mw'].values**2 + b_k*pes.gen['max_p_mw'].values +
                        c_k + a_k*pes.gen['min_p_mw'].values**2 + b_k*pes.gen['min_p_mw'].values + c_k)/2)
        f_k = (4*np.pi/(pes.gen['max_p_mw'].values-pes.gen['min_p_mw'].values))

        a_k_t = np.concatenate((a_k0, np.array(a_k)))
        b_k_t = np.concatenate((b_k0, np.array(b_k)))
        c_k_t = np.concatenate((c_k0, np.array(c_k)))
        e_k_t = np.concatenate((e_k0, np.array(e_k)))
        f_k_t = np.concatenate((f_k0, np.array(f_k)))

        pg_min0 = pes.ext_grid['min_p_mw'].values

        pg_mins = pes.gen['min_p_mw'].values

        pg_min = np.concatenate((pg_min0, pg_mins))

        pg0 = pes.res_ext_grid['p_mw'].values

        pgs = grupo[linha, n_vgen+n_tap +
                    n_bshunt:n_vgen+n_tap+n_bshunt+n_pot]*100

        pg = np.concatenate((pg0, pgs))

        c = np.abs(np.sum((pg**2)*a_k_t+pg*b_k_t+c_k_t +
                   np.abs(e_k_t*np.sin(f_k_t*(pg_min-pg)))))

        grupo[linha, -7] = c

        grupo[linha, -6] = pen_tensao(vbus, v_lim_superior,
                                      v_lim_inferior, relatorio=False)

        vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior, barra = get_generator_data(
            pes, relatorio=False)

        grupo[linha, -5] = pen_ger_reativo(qgen, q_lim_superior,
                                           q_lim_inferior, pes, relatorio=False)

        grupo[linha, :] = pen_trafo(grupo[linha, :], n_tap, n_vgen)

        grupo[linha, -3] = pen_bshunt(grupo[linha, :],
                                      n_tap, n_vgen, n_bshunt, pes)

        # pen gerador 0

        pg_0 = pes.res_ext_grid['p_mw'].values

        pg_0_min = pes.ext_grid['min_p_mw'].values

        pg_0_max = pes.ext_grid['max_p_mw'].values

        pen = 0

        if pg_0 > pg_0_max:

            pen = (pg_0_max-pg_0)**2

        elif pg_0 < pg_0_min:

            pen = (pg_0-pg_0_min)**2

        else:
            pass

        if len(pes.bus) == 30:

            if 55 <= pg_0 <= 66:

                vector = np.array([55, 66])
                posi = np.argmin(np.abs(np.array([55, 66])-pg_0))
                pen = pen + np.abs(vector[posi]-pg_0)

            elif 80 <= pg_0 <= 120:

                vector = np.array([80, 120])
                posi = np.argmin(np.abs(np.array([80, 120])-pg_0))
                pen = pen + np.abs(vector[posi]-pg_0)

            else:
                pass

        if len(pes.bus) == 14:

            if (1/7)*332.4 <= pg_0 <= (2/7)*332.4:

                mean = (((1/7)*332.4)+((2/7)*332.4))/2

                if pg_0 <= mean:

                    pen = pen + np.abs((1/7)*332.4-pg_0)

                else:

                    pen = pen + np.abs((2/7)*332.4-pg_0)

            elif (3/7)*332.4 <= pg_0 <= (4/7)*332.4:

                mean = (((3/7)*332.4)+((4/7)*332.4))/2

                if pg_0 <= mean:

                    pen = pen + np.abs((3/7)*332.4-pg_0)

                else:

                    pen = pen + np.abs((4/7)*332.4-pg_0)

            elif (5/7)*332.4 <= pg_0 <= (6/7)*332.4:

                mean = (((5/7)*332.4)+((6/7)*332.4))/2

                if pg_0 <= mean:

                    pen = pen + np.abs((5/7)*332.4-pg_0)

                else:

                    pen = pen + np.abs((6/7)*332.4-pg_0)

        grupo[linha, -2] = pen

    return grupo


def fitness_fpo(grupo, zeta, psi, sigma, omega, neta):

    # fitness J       perdas/custo/desvio         pen tensão         pen q mvar          pen trafo           pen bshunt
    grupo[:, -1] = (grupo[:, -7])+(zeta*grupo[:, -6])+(psi*grupo[:, -5]) + \
        (sigma*grupo[:, -4])+(omega*grupo[:, -3]) + (neta*grupo[:, -2])

    return grupo


def validacao_fpo(pes, best_solution, relatorio=True):

    valida = fluxo_de_pot_fpo(np.array([best_solution]), pes)

    if relatorio == True:
        print('Função Objetivo:\n')
        print(valida[0][-7])
        print(' ')

        print('Penalização de Violação de Tensão [PU]:\n')
        print(valida[0][-6])
        print(' ')

        print('Penalização de Violação de Geração de Reativo [PU]:\n')
        print(valida[0][-5])
        print(' ')

        print('Penalização de Violação de Geração de Ativo (Slack) [PU]:\n')
        print(valida[0][-2])
        print(' ')

    return valida


def validacao_fpo_q(pes, best_solution, relatorio=True):

    valida = fluxo_de_pot_fpo_q(np.array([best_solution]), pes)

    if relatorio == True:

        print('Função Objetivo:\n')
        print(valida[0][-7])
        print(' ')

        print('Penalização de Violação de Tensão [PU]:\n')
        print(valida[0][-6])
        print(' ')

        print('Penalização de Violação de Geração de Reativo [PU]:\n')
        print(valida[0][-5])
        print(' ')

        print('Penalização de Violação de Geração de Ativo (Slack) [PU]:\n')
        print(valida[0][-2])
        print(' ')

    return valida


def poz(n_vgen, n_tap, n_bshunt, n_gens, particula, pes):

    power_gen = particula[n_vgen+n_tap +
                          n_bshunt:n_vgen+n_tap+n_bshunt+n_gens].copy()*100

    if len(pes.bus) == 14:

        if (1/7)*140 <= power_gen[0] <= (2/7)*140:

            mean = (((1/7)*140)+((2/7)*140))/2

            if power_gen[0] <= mean:

                power_gen[0] = (1/7)*140

            else:

                power_gen[0] = (2/7)*140

        elif (3/7)*140 <= power_gen[0] <= (4/7)*140:

            mean = (((3/7)*140)+((4/7)*140))/2

            if power_gen[0] <= mean:

                power_gen[0] = (3/7)*140

            else:

                power_gen[0] = (4/7)*140

        elif (5/7)*140 <= power_gen[0] <= (6/7)*140:

            mean = (((5/7)*140)+((6/7)*140))/2

            if power_gen[0] <= mean:

                power_gen[0] = (5/7)*140

            else:

                power_gen[0] = (6/7)*140

        if (1/7)*100 <= power_gen[1] <= (2/7)*100:

            mean = (((1/7)*100)+((2/7)*100))/2
            if power_gen[1] <= mean:

                power_gen[1] = (1/7)*100

            else:

                power_gen[1] = (2/7)*100

        elif (3/7)*100 <= power_gen[1] <= (4/7)*100:

            mean = (((3/7)*100)+((4/7)*100))/2

            if power_gen[1] <= mean:

                power_gen[1] = (3/7)*100

            else:

                power_gen[1] = (4/7)*100

        elif (5/7)*100 <= power_gen[1] <= (6/7)*100:

            mean = (((5/7)*100)+((6/7)*100))/2

            if power_gen[1] <= mean:

                power_gen[1] = (5/7)*100

            else:

                power_gen[1] = (6/7)*100

        if (1/7)*100 <= power_gen[2] <= (2/7)*100:

            mean = (((1/7)*100)+((2/7)*100))/2

            if power_gen[2] <= mean:

                power_gen[2] = (1/7)*100

            else:

                power_gen[2] = (2/7)*100

        elif (3/7)*100 <= power_gen[2] <= (4/7)*100:

            mean = (((3/7)*100)+((4/7)*100))/2

            if power_gen[2] <= mean:

                power_gen[2] = (3/7)*100

            else:

                power_gen[2] = (4/7)*100

        elif (5/7)*100 <= power_gen[2] <= (6/7)*100:

            mean = (((5/7)*100)+((6/7)*100))/2

            if power_gen[2] <= mean:

                power_gen[2] = (5/7)*100

            else:

                power_gen[2] = (6/7)*100

        if (1/7)*100 <= power_gen[3] <= (2/7)*100:

            mean = (((1/7)*100)+((2/7)*100))/2

            if power_gen[3] <= mean:

                power_gen[3] = (1/7)*100

            else:

                power_gen[3] = (2/7)*100

        elif (3/7)*100 <= power_gen[3] <= (4/7)*100:

            mean = (((3/7)*100)+((4/7)*100))/2

            if power_gen[3] <= mean:

                power_gen[3] = (3/7)*100

            else:

                power_gen[3] = (4/7)*100

        elif (5/7)*100 <= power_gen[3] <= (6/7)*100:

            mean = (((5/7)*100)+((6/7)*100))/2

            if power_gen[3] <= mean:

                power_gen[3] = (5/7)*100

            else:

                power_gen[3] = (6/7)*100

    if len(pes.bus) == 30:

        # gerador 1

        if 21 <= power_gen[0] <= 24:

            mean = (21+24)/2

            if power_gen[0] <= mean:

                power_gen[0] = 21

            else:

                power_gen[0] = 24

        elif 45 <= power_gen[0] <= 55:

            mean = (45+55)/2

            if power_gen[0] <= mean:

                power_gen[0] = 45

            else:

                power_gen[0] = 55

        # gerador 2

        if 30 <= power_gen[1] <= 36:

            mean = (30+36)/2

            if power_gen[1] <= mean:

                power_gen[1] = 30

            else:

                power_gen[1] = 36

        # gerador 3

        if 25 <= power_gen[2] <= 30:

            mean = (25+30)/2

            if power_gen[2] <= mean:

                power_gen[2] = 25

            else:

                power_gen[2] = 30

        # gerador 4

        if 25 <= power_gen[3] <= 28:

            mean = (25+28)/2

            if power_gen[3] <= mean:

                power_gen[3] = 25

            else:

                power_gen[3] = 28

        # gerador 5

        if 24 <= power_gen[4] <= 30:

            mean = (24+30)/2

            if power_gen[4] <= mean:

                power_gen[4] = 24

            else:

                power_gen[4] = 30


# IEEE 118

    if len(pes.bus) == 118:

        posicoes = [0, 1, 2, 6, 15, 29]

        for val in posicoes:

            if 20 <= power_gen[val] <= 30:

                mean = (20+30)/2

                if power_gen[val] <= mean:

                    power_gen[val] = 20

                else:

                    power_gen[val] = 30

            elif 60 <= power_gen[val] <= 85:

                mean = (60+85)/2

                if power_gen[val] <= mean:

                    power_gen[val] = 60

                else:

                    power_gen[val] = 85

        val = 4

        if 15 <= power_gen[val] <= 45:

            mean = (15+45)/2

            if power_gen[val] <= mean:

                power_gen[val] = 15

            else:

                power_gen[val] = 45

        elif 165 <= power_gen[val] <= 200:

            mean = (165+200)/2

            if power_gen[val] <= mean:

                power_gen[val] = 165

            else:

                power_gen[val] = 200

        elif 395 <= power_gen[val] <= 410:

            mean = (395+410)/2

            if power_gen[val] <= mean:

                power_gen[val] = 395

            else:

                power_gen[val] = 410

        val = 10

        if 40 <= power_gen[val] <= 65:

            mean = (40+65)/2

            if power_gen[val] <= mean:

                power_gen[val] = 40

            else:

                power_gen[val] = 65

        elif 190 <= power_gen[val] <= 200:

            mean = (190+200)/2

            if power_gen[val] <= mean:

                power_gen[val] = 190

            else:

                power_gen[val] = 200

        val = 12

        if 75 <= power_gen[val] <= 95:

            mean = (75+95)/2

            if power_gen[val] <= mean:

                power_gen[val] = 75

            else:

                power_gen[val] = 95

        elif 260 <= power_gen[val] <= 280:

            mean = (260+280)/2

            if power_gen[val] <= mean:

                power_gen[val] = 260

            else:

                power_gen[val] = 280

        val = 20

        if 45 <= power_gen[val] <= 60:

            mean = (45+60)/2

            if power_gen[val] <= mean:

                power_gen[val] = 45

            else:

                power_gen[val] = 60

        elif 185 <= power_gen[val] <= 200:

            mean = (185+200)/2

            if power_gen[val] <= mean:

                power_gen[val] = 185

            else:

                power_gen[val] = 200

        val = 24

        if 95 <= power_gen[val] <= 105:

            mean = (95+105)/2

            if power_gen[val] <= mean:

                power_gen[val] = 95

            else:

                power_gen[val] = 105

        elif 140 <= power_gen[val] <= 155:

            mean = (140+155)/2

            if power_gen[val] <= mean:

                power_gen[val] = 140

            else:

                power_gen[val] = 155

        val = 25

        if 145 <= power_gen[val] <= 155:

            mean = (145+155)/2

            if power_gen[val] <= mean:

                power_gen[val] = 145

            else:

                power_gen[val] = 155

        elif 210 <= power_gen[val] <= 230:

            mean = (210+230)/2

            if power_gen[val] <= mean:

                power_gen[val] = 210

            else:

                power_gen[val] = 230

        val = 27

        if 180 <= power_gen[val] <= 200:

            mean = (200+180)/2

            if power_gen[val] <= mean:

                power_gen[val] = 180

            else:

                power_gen[val] = 200

        elif 350 <= power_gen[val] <= 360:

            mean = (350+360)/2

            if power_gen[val] <= mean:

                power_gen[val] = 350

            else:

                power_gen[val] = 360

        val = 38

        if 120 <= power_gen[val] <= 145:

            mean = (120+145)/2

            if power_gen[val] <= mean:

                power_gen[val] = 120

            else:

                power_gen[val] = 145

        elif 410 <= power_gen[val] <= 460:

            mean = (410+460)/2

            if power_gen[val] <= mean:

                power_gen[val] = 410

            else:

                power_gen[val] = 460

        elif 500 <= power_gen[val] <= 525:

            mean = (500+525)/2

            if power_gen[val] <= mean:

                power_gen[val] = 500

            else:

                power_gen[val] = 525

        posicoes = [17, 18, 36, 42, 45, 52]

        for val in posicoes:

            if 20 <= power_gen[val] <= 30:

                mean = (20+30)/2

                if power_gen[val] <= mean:

                    power_gen[val] = 20

                else:

                    power_gen[val] = 30

            elif 45 <= power_gen[val] <= 55:

                mean = (45+55)/2

                if power_gen[val] <= mean:

                    power_gen[val] = 45

                else:

                    power_gen[val] = 55

    p_return = particula.copy()

    p_return[n_vgen+n_tap+n_bshunt:n_vgen +
             n_tap+n_bshunt+n_gens] = power_gen/100

    return p_return
