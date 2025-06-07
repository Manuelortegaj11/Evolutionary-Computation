# Funciones para obtener la información que requiere el modelo, y definición de función objetivo
import numpy as np
import pandas as pd


# Información que obtiene el modelo del sistema de información
def model_data(files, demanda, ctiempo, t_max=360, folder='./data/'):
    # Demanda
    d = demanda
    # Costo por unidad de tiempo
    ct = ctiempo
    # Tiempo máximo definido
    t = t_max
    # Parámetros de entrada
    pdf = pd.read_excel(folder + files['info_acopios'])
    # Matriz de costos de transporte
    tcdf = pd.read_excel(folder + files['costo_transporte'], index_col=0)
    # Matriz de tiempos de transporte
    ttdf = pd.read_excel(folder + files['tiempo_transporte'], index_col=0)
    
    data = {
        'demanda': d,
        'ctiempo': ct,
        't_max': t,
        'params_df': pdf,
        'trans_costo_df': tcdf,
        'trans_tiempo_df': ttdf
    }
    
    return data


# Definición de variables para el modelo
def model_vars(params_df, seed=1):
    # Número de centros de acopio
    n = params_df.shape[0]
    # Vector de representación
    cap = np.empty(n*2, dtype=float)
    
    # Capacidades en stock y potencial por centro de acopio
    for cap_i in range(0, n*2, 2):
        cap[cap_i] = params_df['Stock'].iloc[cap_i//2]
        cap[cap_i+1] = params_df['Ppotencial'].iloc[cap_i//2]
    
    cap = np.append(cap, n-1)
    s = seed
    
    return n, s, cap


# Función para obtener la suma de costos, usada por la función objetivo
def __get_delta(x, i, model_dict, idx_acopio, idx_principal=-1):
    ctiempo = model_dict['ctiempo']
    params_df = model_dict['params_df']
    trans_costo_df = model_dict['trans_costo_df']
    trans_tiempo_df = model_dict['trans_tiempo_df']
    
    kca = x[i] + x[i + 1]
    precio = params_df['Precio'].iloc[idx_acopio]
    talistam = 0

    if x[i + 1]:
        talistam = params_df['TiempoAlistam'].iloc[idx_acopio]

    ' Si es el centro de acopio principal (idx_principal=-1) '
    if idx_principal < 0:
        ctransp = params_df['Ctransp'].iloc[idx_acopio]
        ttransp = params_df['TiempoTransp'].iloc[idx_acopio]
        tiempo = talistam + ttransp

        return (kca * precio) + ctransp + (tiempo * ctiempo)
    else:
        ctransp = trans_costo_df.iloc[idx_acopio, idx_principal]
        ttransp = trans_tiempo_df.iloc[idx_acopio, idx_principal]
        tiempo = talistam + ttransp

        return (kca * precio) + ctransp + (tiempo * ctiempo)


# Función objetivo
def objective_func(x, n, model_dict):
    delta = 0
    idx_principal = int(x[n*2])

    for i in range(0, n*2, 2):
        idx_acopio = i//2

        if x[i] == 0 and x[i+1] == 0:
            continue

        if idx_acopio == idx_principal:
            # Única suma de kCAp
            delta += __get_delta(x, i, model_dict, idx_acopio)
            continue

        # Sumas con kCAi
        delta += __get_delta(x, i, model_dict, idx_acopio, idx_principal)

    return delta


# Función que altera el vector para que cumpla con la restricción de igualdad
def balance(vector, cap, delta, diff):
    delta = np.squeeze(delta)

    # Reducir para igualar a la demanda
    if diff:
        acopios = list(np.nonzero(vector)[0])

        while delta > 0 and len(acopios) > 0:
            idx = np.random.choice(acopios)
            acopios.remove(idx)

            if delta <= vector[idx]:
                vector[idx] -= delta
                delta = 0
            else:
                delta -= vector[idx]
                vector[idx] = 0
    # Aumentar para igualar a la demanda
    else:
        acopios = list(np.where(vector == 0)[0])

        while delta > 0 and len(acopios) > 0:
            idx = np.random.choice(acopios)
            acopios.remove(idx)

            if delta <= cap[idx]:
                vector[idx] = delta
                delta = 0
            else:
                vector[idx] = cap[idx]
                delta -= cap[idx]


# Crea una tabla que muestra la asignación de cantidades asignadas por centro de acopio (stock/potencial)
def alloc_df(x, params_df, n, cap, sto='Stock', pot='Potencial'):
    x = np.delete(x, n*2)
    cap = np.delete(cap, n*2)
    size = len(x)

    c_evens = np.take(cap, [idx for idx in range(0, size, 2)])
    c_odds = np.take(cap, [idx for idx in range(1, size, 2)])
    evens = np.take(x, [idx for idx in range(0, size, 2)])
    odds = np.take(x, [idx for idx in range(1, size, 2)])

    data_dict = {
        'CAcopio': params_df['Id_CA'],
        'C.Stock': c_evens,
        sto: evens,
        'C.Potencial': c_odds,
        pot: odds,
    }
    ca_df = pd.DataFrame.from_dict(data_dict)
    return ca_df
