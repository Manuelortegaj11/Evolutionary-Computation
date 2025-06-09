import numpy as np
import pandas as pd
import timeit
import os
from joblib import Parallel, delayed
from scripts.queso_model import (
    model_vars,
    model_data,
    objective_func,
    balance,
    alloc_df,
)
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize

# Determinar la raíz del proyecto y construir rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_base = os.path.join(project_root, "bio-inspired methods", "data")
results_base = os.path.join(
    project_root,
    "bio-inspired methods",
    "genetic-algorithm✅",
    "experimentation and testing",
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


# Clases y funciones originales
class Queso(ElementwiseProblem):
    def __init__(self, data, N, demanda):
        super().__init__(n_var=len(capacidades), n_obj=1, n_eq_constr=1, xl=xl, xu=xu)
        self.data = data
        self.N = N
        self.demanda = demanda

    def _evaluate(self, x, out):
        out["F"] = objective_func(x, self.N, self.data)
        individual = np.delete(x, self.N * 2)
        out["H"] = self.demanda - np.sum(individual)


class TopOrZeroSampling(Sampling):
    def __init__(self, capacidades, demanda, N):
        super().__init__()
        self.capacidades = capacidades
        self.demanda = demanda
        self.N = N

    def _do(self, problem, n_samples, **kwargs):
        gen_matrix = np.zeros((n_samples, problem.n_var), dtype=float)
        n_vars = self.N * 2 + 1
        for i in range(n_samples):
            indices = np.arange(n_vars - 1)
            np.random.shuffle(indices)
            while np.sum(gen_matrix[i]) < self.demanda and indices.size > 0:
                idx = indices[0]
                gen_matrix[i, idx] = self.capacidades[idx]
                indices = np.delete(indices, 0)
                if np.sum(gen_matrix[i]) > self.demanda:
                    gen_matrix[i, idx] = gen_matrix[i, idx] - (
                        np.sum(gen_matrix[i]) - self.demanda
                    )
                    break
            gen_matrix[i, self.N * 2] = np.random.randint(
                self.capacidades[self.N * 2] + 1
            )
        return gen_matrix


class SinglePointCross(Crossover):
    def __init__(self, prob, capacidades, demanda):
        super().__init__(n_parents=2, n_offsprings=1, prob=prob)
        self.capacidades = capacidades
        self.demanda = demanda

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        T = np.zeros((1, n_matings, n_var))
        Y = np.full_like(T, None, dtype=float)
        for idx in range(n_matings):
            p1 = X[0, idx, : n_var // 2]
            p2 = X[1, idx, n_var // 2 : n_var - 1]
            offspring = np.concatenate((p1, p2))
            if np.sum(offspring) > self.demanda:
                delta = np.sum(offspring) - self.demanda
                balance(offspring, self.capacidades, delta, True)
            else:
                delta = self.demanda - np.sum(offspring)
                balance(offspring, self.capacidades, delta, False)
            for i in range(offspring.shape[0]):
                Y[0, idx, i] = offspring[i]
            main = np.random.choice([X[0, idx, n_var - 1], X[1, idx, n_var - 1]])
            Y[0, idx, n_var - 1] = main
        return Y


class ReassignMutation(Mutation):
    def __init__(self, prob, capacidades, demanda, N):
        super().__init__()
        self.prob = prob
        self.capacidades = capacidades
        self.demanda = demanda
        self.N = N

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            r = np.random.random()
            if r < self.prob:
                individual = X[i]
                idx_mut = np.random.randint(individual.shape[0])
                if idx_mut == self.N * 2:
                    X[i, problem.n_var - 1] = np.random.randint(
                        self.capacidades[self.N * 2] + 1
                    )
                else:
                    if individual[idx_mut] == 0:
                        delta = self.capacidades[idx_mut]
                        individual[idx_mut] = self.capacidades[idx_mut]
                        diff = True
                    else:
                        delta = individual[idx_mut]
                        individual[idx_mut] = 0
                        diff = False
                    balance(individual, self.capacidades, delta, diff)
                    for j in range(individual.shape[0]):
                        X[i, j] = individual[j]
        return X


def run_experiment(
    p_mutate,
    p_cross,
    pop_size,
    max_gen,
    block_name,
    data,
    N,
    capacidades,
    xl,
    xu,
    demanda,
    runs=100,
):
    def single_run(run, p_mutate, p_cross, pop_size, max_gen):
        t_start = timeit.default_timer()
        algorithm = GA(
            pop_size=pop_size,
            sampling=TopOrZeroSampling(capacidades, demanda, N),
            crossover=SinglePointCross(
                prob=p_cross, capacidades=capacidades, demanda=demanda
            ),
            mutation=ReassignMutation(
                prob=p_mutate, capacidades=capacidades, demanda=demanda, N=N
            ),
            eliminate_duplicates=True,
        )
        termination = DefaultSingleObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=1e-6,
            period=100,
            n_max_gen=1000000,
            n_max_evals=100000,
        )
        ga = minimize(
            Queso(data, N, demanda),
            algorithm,
            termination,
            save_history=False,
            verbose=False,
        )
        t_end = timeit.default_timer()
        costo = np.squeeze(ga.F) if ga.F is not None else float("inf")
        iterations = ga.history[-1].n_gen if ga.history else 0
        return {
            "block": block_name,
            "T_mutacion": p_mutate,
            "T_cruce": p_cross,
            "size_pob": pop_size,
            "N_gen": max_gen,
            "run": run + 1,
            "costo": costo,
            "tiempo": t_end - t_start,
        }

    results = Parallel(n_jobs=4)(
        delayed(single_run)(i, p_mutate, p_cross, pop_size, max_gen)
        for i in range(runs)
    )
    return results


def summarize_experiment(results):
    df = pd.DataFrame(results)
    summary = (
        df.groupby(["block", "T_mutacion", "T_cruce", "size_pob", "N_gen"])
        .agg(mean=("costo", "mean"), std=("costo", "std"), avg_time=("tiempo", "mean"))
        .reset_index()
    )
    summary["var.coeff"] = summary["std"] / summary["mean"]
    summary = summary[
        [
            "block",
            "T_mutacion",
            "T_cruce",
            "size_pob",
            "N_gen",
            "mean",
            "var.coeff",
            "avg_time",
        ]
    ]
    return summary


# Parámetros para los experimentos
block1_p_mutate = [0.5, 0.7, 0.9]
block1_p_cross = [0.5, 0.7, 0.9]
block1_pop_size = [50, 100, 150]
block1_max_gen = [100, 300, 500]

block2_p_mutate = np.arange(0.1, 0.6, 0.1).tolist()
block2_p_cross = np.arange(0.1, 0.8, 0.1).tolist()
block2_pop_size = block1_pop_size
block2_max_gen = block1_max_gen

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
    info_acopios = "centros_acopio.xlsx"
    costo_transporte = "costos_transporte.xlsx"
    tiempo_transporte = "tiempos_transporte.xlsx"
    archivos = {
        "info_acopios": info_acopios,
        "costo_transporte": costo_transporte,
        "tiempo_transporte": tiempo_transporte,
    }
    missing_files = []
    for file in [os.path.join(folder, f) for f in archivos.values()]:
        if not os.path.exists(file):
            missing_files.append(os.path.basename(file))
    if missing_files:
        print(
            f"Error: Archivos no encontrados en {folder}: {missing_files}. Saltando escenario {i + 1}."
        )
        continue

    # Cargar datos para el escenario
    try:
        data = model_data(archivos, demanda, folder=folder)
        N, _, capacidades = model_vars(data["params_df"])
        xl = np.zeros(capacidades.shape[0])
        xu = capacidades
    except Exception as e:
        print(f"Error al cargar datos en {folder}: {e}. Saltando escenario {i + 1}.")
        continue

    # Iniciar medición de tiempo para el escenario
    start_time = timeit.default_timer()
    total_combinations = (
        len(block1_p_mutate)
        * len(block1_p_cross)
        * len(block1_pop_size)
        * len(block1_max_gen)
    ) + (
        len(block2_p_mutate)
        * len(block2_p_cross)
        * len(block2_pop_size)
        * len(block2_max_gen)
    )
    processed_combinations = 0

    # Ejecutar experimentos
    all_results = []
    for p_mutate in block1_p_mutate:
        for p_cross in block1_p_cross:
            for pop_size in block1_pop_size:
                for max_gen in block1_max_gen:
                    all_results.extend(
                        run_experiment(
                            p_mutate,
                            p_cross,
                            pop_size,
                            max_gen,
                            "Bloque 1",
                            data,
                            N,
                            capacidades,
                            xl,
                            xu,
                            demanda,
                        )
                    )
                    processed_combinations += 1
                    elapsed_time = timeit.default_timer() - start_time
                    avg_time_per_comb = elapsed_time / processed_combinations
                    remaining_combinations = total_combinations - processed_combinations
                    remaining_time = avg_time_per_comb * remaining_combinations
                    print(
                        f"Escenario {i + 1}: Procesadas {processed_combinations}/{total_combinations} combinaciones. Faltan aproximadamente {remaining_time / 60:.1f} minutos para terminar."
                    )

    for p_mutate in block2_p_mutate:
        for p_cross in block2_p_cross:
            for pop_size in block2_pop_size:
                for max_gen in block2_max_gen:
                    all_results.extend(
                        run_experiment(
                            p_mutate,
                            p_cross,
                            pop_size,
                            max_gen,
                            "Bloque 2",
                            data,
                            N,
                            capacidades,
                            xl,
                            xu,
                            demanda,
                        )
                    )
                    processed_combinations += 1
                    elapsed_time = timeit.default_timer() - start_time
                    avg_time_per_comb = elapsed_time / processed_combinations
                    remaining_combinations = total_combinations - processed_combinations
                    remaining_time = avg_time_per_comb * remaining_combinations
                    print(
                        f"Escenario {i + 1}: Procesadas {processed_combinations}/{total_combinations} combinaciones. Faltan aproximadamente {remaining_time / 60:.1f} minutos para terminar."
                    )

    # Guardar resultados
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(resultados_folder, "experiment_results_ga.xlsx")
    try:
        results_df.to_excel(results_file, index=False)
        print(
            f"Resultados exportados a '{results_file}' con columnas: block, T_mutacion, T_cruce, size_pob, N_gen, run, costo, tiempo"
        )
    except Exception as e:
        print(f"Error al exportar '{results_file}': {e}")

    summary_df = summarize_experiment(all_results)
    summary_file = os.path.join(resultados_folder, "experiment_summary_ga.xlsx")
    try:
        summary_df.to_excel(summary_file, index=False)
        print(
            f"Resumen exportado a '{summary_file}' con columnas: block, T_mutacion, T_cruce, size_pob, N_gen, mean, var.coeff, avg_time"
        )
    except Exception as e:
        print(f"Error al exportar '{summary_file}': {e}")
