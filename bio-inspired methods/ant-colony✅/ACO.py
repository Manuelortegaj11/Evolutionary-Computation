import numpy as np
import pandas as pd
import copy
import timeit
import os
from joblib import Parallel, delayed

# Determinar la raíz del proyecto y construir rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_base = os.path.join(project_root, "bio-inspired methods", "data")
results_base = os.path.join(
    project_root, "bio-inspired methods", "ant-colony✅", "experimentation and testing"
)

# Definir los cuatro escenarios con sus carpetas y demandas
escenarios = [
    {
        # Aproximadamente el 26% de la capacidad total (232.625), adecuada para un escenario pequeño.
        "folder": os.path.join(data_base, "escenario_1/"),
        "demanda": 60,
        "resultados_folder": os.path.join(results_base, "resultados_escenario_1/"),
    },
    {
        # Aproximadamente el 78% de la capacidad total (15339), refleja la alta capacidad disponible.
        "folder": os.path.join(data_base, "escenario_2/"),
        "demanda": 12000,
        "resultados_folder": os.path.join(results_base, "resultados_escenario_2/"),
    },
    {
        # Aproximadamente el 70% de la capacidad total (114), balancea la baja disponibilidad de stock y potencial.
        "folder": os.path.join(data_base, "escenario_3/"),
        "demanda": 80,
        "resultados_folder": os.path.join(results_base, "resultados_escenario_3/"),
    },
    {
        # Aproximadamente el 86% de la capacidad total (58366), desafía a los algoritmos con alta demanda y muchos centros.
        "folder": os.path.join(data_base, "escenario_4/"),
        "demanda": 50000,
        "resultados_folder": os.path.join(results_base, "resultados_escenario_4/"),
    },
]

feromona_inicial = 1 / 100
Q = 1000


def load_data(folder):
    info_CA = os.path.join(folder, "centros_acopio.xlsx")
    info_costTransp = os.path.join(folder, "costos_transporte.xlsx")
    info_timeTransp = os.path.join(folder, "tiempos_transporte.xlsx")

    df_infoCA = pd.read_excel(info_CA)
    df_infoCA.set_index(df_infoCA.columns[0], inplace=True)
    df_cTransp = pd.read_excel(info_costTransp)
    df_cTransp.set_index(df_cTransp.columns[0], inplace=True)
    df_tTransp = pd.read_excel(info_timeTransp)
    df_tTransp.set_index(df_tTransp.columns[0], inplace=True)

    ctiempo = df_infoCA["Precio"] * 0.1  # ctiempo ahora se calcula como 10% del precio.
    return df_infoCA, df_cTransp, df_tTransp, ctiempo


class CA:
    def __init__(self, idCA, cant_stock, cant_pot):
        self.idCA = idCA
        self.en_Stock = cant_stock
        self.potencial = cant_pot


class Ant:
    def __init__(self, demanda, CAs_list, i, Q):
        self.ant_id = i
        self.sol_CAs = []
        self.cheese_cant = 0
        self.available_CAs = copy.deepcopy(CAs_list)
        self.demanda = demanda
        self.CAP = None
        self.idx_CAP = 0
        self.eta_CAP = 0
        self.costo_total = 0
        self.pheromone_delta = np.zeros(len(self.available_CAs))
        self.Q = Q
        self.tao_ini = Q / Q

    def select_CAP(self, df_infoCA, costo_tiempo, alpha, beta):
        deltica = 0.00001
        tao = self.tao_ini
        temp = np.zeros(len(self.available_CAs))
        denominador = 0
        for i, CA in enumerate(self.available_CAs):
            if CA.en_Stock > 0:
                to_take = min(CA.en_Stock, self.demanda)
                eta = (
                    to_take * df_infoCA.loc[CA.idCA, "Precio"]
                    + df_infoCA.loc[CA.idCA, "Ctransp"]
                    + df_infoCA.loc[CA.idCA, "TiempoTransp"] * costo_tiempo.iloc[i]
                )
                eta_beta = eta**beta
                tao_alpha = tao**alpha
                temp[i] = eta_beta * tao_alpha
                denominador += temp[i]
        probabilities = np.divide(temp, denominador + deltica)
        ids = np.argmax(probabilities)
        self.CAP = self.available_CAs[ids].idCA
        self.cheese_cant = min(self.available_CAs[ids].en_Stock, self.demanda)
        self.available_CAs[ids].en_Stock -= self.cheese_cant
        self.idx_CAP = ids
        self.sol_CAs.append((self.CAP, self.cheese_cant, "Stock"))
        self.eta_CAP = (
            self.cheese_cant * df_infoCA.loc[self.CAP, "Precio"]
            + df_infoCA.loc[self.CAP, "Ctransp"]
            + df_infoCA.loc[self.CAP, "TiempoTransp"] * costo_tiempo.iloc[ids]
        )
        self.costo_total = self.eta_CAP

    def select_next_CA(
        self, df_infoCA, df_cTransp, df_tTransp, costo_tiempo, alpha, beta, pheromone
    ):
        prob_temp = np.zeros(len(self.available_CAs))
        denominador = 0
        for idx, CA in enumerate(self.available_CAs):
            if CA.en_Stock + CA.potencial > 0:
                if CA.en_Stock > 0:
                    to_take = min(CA.en_Stock, self.demanda - self.cheese_cant)
                    eta = (
                        to_take * df_infoCA.loc[CA.idCA, "Precio"]
                        + df_cTransp.loc[self.CAP, CA.idCA]
                        + df_tTransp.loc[self.CAP, CA.idCA] * costo_tiempo.iloc[idx]
                    )
                elif CA.potencial > 0:
                    to_take = min(CA.potencial, self.demanda - self.cheese_cant)
                    eta = (
                        to_take * df_infoCA.loc[CA.idCA, "Precio"]
                        + df_infoCA.loc[CA.idCA, "TiempoAlistam"]
                        * costo_tiempo.iloc[idx]
                    )
                eta_beta = (eta + self.costo_total) ** beta
                tao_alpha = pheromone[self.idx_CAP][idx] ** alpha
                prob_temp[idx] = eta_beta * tao_alpha
                denominador += prob_temp[idx]
        if denominador == 0:
            return -1
        probabilities = np.divide(prob_temp, denominador)
        ids = np.random.choice(len(self.available_CAs), p=probabilities)
        if self.available_CAs[ids].en_Stock + self.available_CAs[ids].potencial == 0:
            return -1
        taken = 0
        tipo = ""
        CA = self.available_CAs[ids].idCA
        if self.available_CAs[ids].en_Stock > 0:
            tipo = "Stock"
            taken = min(
                self.available_CAs[ids].en_Stock, self.demanda - self.cheese_cant
            )
            self.cheese_cant += taken
            self.available_CAs[ids].en_Stock -= taken
            costo = (
                taken * df_infoCA.loc[CA, "Precio"]
                + df_cTransp.loc[self.CAP, CA]
                + df_tTransp.loc[self.CAP, CA] * costo_tiempo.iloc[ids]
            )
        elif self.available_CAs[ids].potencial > 0:
            tipo = "Potencial"
            taken = min(
                self.available_CAs[ids].potencial, self.demanda - self.cheese_cant
            )
            self.cheese_cant += taken
            self.available_CAs[ids].potencial -= taken
            costo = (
                taken * df_infoCA.loc[CA, "Precio"]
                + df_infoCA.loc[CA, "TiempoAlistam"] * costo_tiempo.iloc[ids]
            )
        if taken > 0:
            self.costo_total += costo
            self.sol_CAs.append((CA, taken, tipo))
            self.pheromone_delta[ids] += self.Q / costo
        return ids

    def complet_order(
        self, df_infoCA, df_cTransp, df_tTransp, costo_tiempo, alpha, beta, pheromone
    ):
        while self.cheese_cant < self.demanda:
            result = self.select_next_CA(
                df_infoCA, df_cTransp, df_tTransp, costo_tiempo, alpha, beta, pheromone
            )
            if result == -1:
                break


class ACO_cheese:
    def __init__(self, demanda_g, colony_size, feromona_inicial, alpha, beta, rho, Q):
        self.df_infoCA = None
        self.df_cTransp = None
        self.df_tTransp = None
        self.demanda_g = demanda_g
        self.N_CA = 0
        self.costo_tiempo = None
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.colony_size = colony_size
        self.disponibles = []
        self.pheromone = []
        self.feromona_inicial = feromona_inicial

    def read_infoProblem(self, df_infoCA, df_cTransp, df_tTransp, costo_tiempo):
        self.df_infoCA = df_infoCA
        self.N_CA = len(self.df_infoCA)
        self.df_cTransp = df_cTransp
        self.df_tTransp = df_tTransp
        self.costo_tiempo = costo_tiempo
        self.crea_lista_disponibles()

    def crea_lista_disponibles(self):
        for index, r in self.df_infoCA.iterrows():
            oferta1 = CA(index, r["Stock"], r["Ppotencial"])
            self.disponibles.append(oferta1)
        self.pheromone = [
            [self.feromona_inicial for _ in range(self.N_CA)] for _ in range(self.N_CA)
        ]

    def update_pheromone(self, colony):
        for i in range(self.N_CA):
            for j in range(self.N_CA):
                self.pheromone[i][j] *= self.rho
        for ant in colony:
            i = ant.idx_CAP
            for j in range(self.N_CA):
                self.pheromone[i][j] += ant.pheromone_delta[j]

    def ACO_run(self, max_iter, stagnation_threshold=50, cost_tol=1e-6):
        N_Ants = self.colony_size
        best_costo = float("inf")
        best_sol = []
        stagnation_count = 0
        prev_best_costo = float("inf")
        iteracion = 0
        while stagnation_count < stagnation_threshold and iteracion < max_iter:
            colony = []
            for i in range(N_Ants):
                a = Ant(self.demanda_g, self.disponibles, i, self.Q)
                a.select_CAP(self.df_infoCA, self.costo_tiempo, self.alpha, self.beta)
                a.complet_order(
                    self.df_infoCA,
                    self.df_cTransp,
                    self.df_tTransp,
                    self.costo_tiempo,
                    self.alpha,
                    self.beta,
                    self.pheromone,
                )
                colony.append(a)
                if a.costo_total < best_costo:
                    best_costo = a.costo_total
                    best_sol = a.sol_CAs
            self.update_pheromone(colony)
            if abs(prev_best_costo - best_costo) < cost_tol:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_best_costo = best_costo
            iteracion += 1
        return best_costo, best_sol, iteracion


def run_experiment(
    rho,
    colony_size,
    max_iter,
    alpha,
    beta,
    block_name,
    df_infoCA,
    df_cTransp,
    df_tTransp,
    ctiempo,
    demanda,
    runs=100,
):
    def single_run(run, rho, colony_size, max_iter, alpha, beta):
        t_start = timeit.default_timer()
        aco = ACO_cheese(demanda, colony_size, feromona_inicial, alpha, beta, rho, Q)
        aco.read_infoProblem(df_infoCA, df_cTransp, df_tTransp, ctiempo)
        costo, _, iterations = aco.ACO_run(max_iter)
        t_end = timeit.default_timer()
        return {
            "block": block_name,
            "T_evaporacion": rho,
            "size_col": colony_size,
            "N_gen": max_iter,
            "alpha": alpha,
            "beta": beta,
            "run": run + 1,
            "costo": costo,
            "tiempo": t_end - t_start,
            "iteraciones": iterations,
        }

    results = Parallel(n_jobs=4)(
        delayed(single_run)(i, rho, colony_size, max_iter, alpha, beta)
        for i in range(runs)
    )
    return results


def summarize_experiment(results):
    df = pd.DataFrame(results)
    summary = (
        df.groupby(["block", "T_evaporacion", "size_col", "N_gen", "alpha", "beta"])
        .agg(
            mean=("costo", "mean"),
            std=("costo", "std"),
            avg_time=("tiempo", "mean"),
            avg_iterations=("iteraciones", "mean"),
        )
        .reset_index()
    )
    summary["var.coeff"] = summary["std"] / summary["mean"]
    summary = summary[
        [
            "block",
            "T_evaporacion",
            "size_col",
            "N_gen",
            "alpha",
            "beta",
            "mean",
            "var.coeff",
            "avg_time",
            "avg_iterations",
        ]
    ]
    return summary


# Parámetros para los experimentos
block1_rho = [0.5, 0.7, 0.9]
block1_colony_size = [20, 30, 50]
block1_max_iter = [50, 100, 150]
block1_alpha = [1.0, 1.5, 2.0]
block1_beta = [1.0, 1.5, 2.0]

block2_rho = np.arange(0.900, 1.000, 0.011).tolist()
block2_colony_size = block1_colony_size
block2_max_iter = block1_max_iter
block2_alpha = block1_alpha
block2_beta = block1_beta

# Ejecutar experimentos para cada escenario
for i, escenario in enumerate(escenarios):
    folder = escenario["folder"]
    demanda = escenario["demanda"]
    resultados_folder = escenario["resultados_folder"]

    # Crear carpeta de resultados si no existe
    os.makedirs(resultados_folder, exist_ok=True)

    # Verificar si la carpeta de datos existe
    if not os.path.exists(folder):
        print(f"Error: La carpeta {folder} no existe. Saltando escenario {i + 1}.")
        continue

    # Verificar la existencia de los archivos
    info_CA = os.path.join(folder, "centros_acopio.xlsx")
    info_costTransp = os.path.join(folder, "costos_transporte.xlsx")
    info_timeTransp = os.path.join(folder, "tiempos_transporte.xlsx")
    missing_files = []
    for file in [info_CA, info_costTransp, info_timeTransp]:
        if not os.path.exists(file):
            missing_files.append(os.path.basename(file))
    if missing_files:
        print(
            f"Error: Archivos no encontrados en {folder}: {missing_files}. Saltando escenario {i + 1}."
        )
        continue

    # Cargar datos para el escenario
    try:
        df_infoCA, df_cTransp, df_tTransp, ctiempo = load_data(folder)
    except Exception as e:
        print(f"Error al cargar datos en {folder}: {e}. Saltando escenario {i + 1}.")
        continue

    # Iniciar medición de tiempo para el escenario
    start_time = timeit.default_timer()
    total_combinations = (
        len(block1_rho)
        * len(block1_colony_size)
        * len(block1_max_iter)
        * len(block1_alpha)
        * len(block1_beta)
    ) + (
        len(block2_rho)
        * len(block2_colony_size)
        * len(block2_max_iter)
        * len(block2_alpha)
        * len(block2_beta)
    )
    processed_combinations = 0

    # Ejecutar experimentos
    all_results = []
    for rho in block1_rho:
        for col_size in block1_colony_size:
            for max_iter in block1_max_iter:
                for alpha in block1_alpha:
                    for beta in block1_beta:
                        all_results.extend(
                            run_experiment(
                                rho,
                                col_size,
                                max_iter,
                                alpha,
                                beta,
                                "Bloque 1",
                                df_infoCA,
                                df_cTransp,
                                df_tTransp,
                                ctiempo,
                                demanda,
                            )
                        )
                        processed_combinations += 1
                        elapsed_time = timeit.default_timer() - start_time
                        avg_time_per_comb = elapsed_time / processed_combinations
                        remaining_combinations = (
                            total_combinations - processed_combinations
                        )
                        remaining_time = avg_time_per_comb * remaining_combinations
                        print(
                            f"Escenario {i + 1}: Procesadas {processed_combinations}/{total_combinations} combinaciones. Faltan aproximadamente {remaining_time / 60:.1f} minutos para terminar."
                        )

    for rho in block2_rho:
        for col_size in block2_colony_size:
            for max_iter in block2_max_iter:
                for alpha in block2_alpha:
                    for beta in block2_beta:
                        all_results.extend(
                            run_experiment(
                                rho,
                                col_size,
                                max_iter,
                                alpha,
                                beta,
                                "Bloque 2",
                                df_infoCA,
                                df_cTransp,
                                df_tTransp,
                                ctiempo,
                                demanda,
                            )
                        )
                        processed_combinations += 1
                        elapsed_time = timeit.default_timer() - start_time
                        avg_time_per_comb = elapsed_time / processed_combinations
                        remaining_combinations = (
                            total_combinations - processed_combinations
                        )
                        remaining_time = avg_time_per_comb * remaining_combinations
                        print(
                            f"Escenario {i + 1}: Procesadas {processed_combinations}/{total_combinations} combinaciones. Faltan aproximadamente {remaining_time / 60:.1f} minutos para terminar."
                        )

    # Guardar resultados
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(resultados_folder, "experiment_results_aco.xlsx")
    try:
        results_df.to_excel(results_file, index=False)
        print(
            f"Resultados exportados a '{results_file}' con columnas: block, T_evaporacion, size_col, N_gen, alpha, beta, run, costo, tiempo, iteraciones"
        )
    except Exception as e:
        print(f"Error al exportar '{results_file}': {e}")

    summary_df = summarize_experiment(all_results)
    summary_file = os.path.join(resultados_folder, "experiment_summary_aco.xlsx")
    try:
        summary_df.to_excel(summary_file, index=False)
        print(
            f"Resumen exportado a '{summary_file}' con columnas: block, T_evaporacion, size_col, N_gen, alpha, beta, mean, var.coeff, avg_time, avg_iterations"
        )
    except Exception as e:
        print(f"Error al exportar '{summary_file}': {e}")
