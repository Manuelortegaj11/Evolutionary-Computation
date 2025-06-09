import numpy as np
import pandas as pd

def model_data(files, demanda, t_max=360, folder='./data/'):

    d = demanda
    
    t = t_max
    
    pdf = pd.read_excel(folder + files['info_acopios'])
    
    ct = pdf['Precio'] * 0.1
    
    tcdf = pd.read_excel(folder + files['costo_transporte'], index_col=0)
    
    ttdf = pd.read_excel(folder + files['tiempo_transporte'], index_col=0)
    
    data = {
        'demanda': d,
        'ctiempo': ct,  # Ahora es una Serie de Pandas con un valor por centro de acopio
        't_max': t,
        'params_df': pdf,
        'trans_costo_df': tcdf,
        'trans_tiempo_df': ttdf
    }
    
    return data

def model_vars(params_df, seed=1):

    n = params_df.shape[0]

    cap = np.empty(n*2, dtype=float)
    

    for cap_i in range(0, n*2, 2):
        cap[cap_i] = params_df['Stock'].iloc[cap_i//2]
        cap[cap_i+1] = params_df['Ppotencial'].iloc[cap_i//2]
    
    cap = np.append(cap, n-1)
    s = seed
    
    return n, s, cap

def __get_delta(x, i, model_dict, idx_acopio, idx_principal=-1):
    ctiempo = model_dict['ctiempo'][idx_acopio]  
    params_df = model_dict['params_df']
    trans_costo_df = model_dict['trans_costo_df']
    trans_tiempo_df = model_dict['trans_tiempo_df']
    
    kca = x[i] + x[i + 1]
    precio = params_df['Precio'].iloc[idx_acopio]
    talistam = 0

    if x[i + 1]:
        talistam = params_df['TiempoAlistam'].iloc[idx_acopio]


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

def objective_func(x, n, model_dict):
    delta = 0
    idx_principal = int(x[n*2])

    for i in range(0, n*2, 2):
        idx_acopio = i//2

        if x[i] == 0 and x[i+1] == 0:
            continue

        if idx_acopio == idx_principal:
            delta += __get_delta(x, i, model_dict, idx_acopio)
            continue

        delta += __get_delta(x, i, model_dict, idx_acopio, idx_principal)

    return delta

def balance(vector, cap, delta, diff):
    delta = np.squeeze(delta)

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
