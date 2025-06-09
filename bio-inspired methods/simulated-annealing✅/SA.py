import pandas as pd
import numpy as np
import timeit
import os
from scripts.queso_model import (
    model_data,
    model_vars,
    objective_func,
    balance,
)
from joblib import Parallel, delayed

# Determinar la raíz del proyecto y construir rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_base = os.path.join(project_root, "bio-inspired methods", "data")
results_base = os.path.join(
    project_root,
    "bio-inspired methods",
    "simulated-annealing✅",
    "experimentation and testing",
)

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

info_acopios = "centros_acopio.xlsx"
costo_transporte = "costos_transporte.xlsx"
tiempo_transporte = "tiempos_transporte.xlsx"

archivos = {
    "info_acopios": info_acopios,
    "costo_transporte": costo_transporte,
    "tiempo_transporte": tiempo_transporte,
}


def gen_s0(n_vars, capacidades, demanda):
    gen_vector = np.zeros(n_vars + 1, dtype=float)
    indices = np.arange(n_vars)
    np.random.shuffle(indices)
    while np.sum(gen_vector) < demanda and indices.size > 0:
        idx = indices[0]
        gen_vector[idx] += capacidades[idx]
        indices = np.delete(indices, 0)
        if np.sum(gen_vector) > demanda:
            gen_vector[idx] -= np.sum(gen_vector) - demanda
            break
    gen_vector[n_vars] = np.random.randint(capacidades[n_vars] + 1)
    return gen_vector


def gen_s(sol, N, capacidades, demanda):
    indices = np.arange(N * 2 + 1)
    np.random.shuffle(indices)
    idx = indices[0]
    while indices.size > 0:
        if idx == N * 2:
            cap = sol[idx].astype(int)
            cap = np.delete(np.arange(capacidades[N * 2] + 1), cap)
            sol[idx] = np.random.choice(cap)
            break
        if sol[idx] == 0 and capacidades[idx] == 0:
            indices = np.delete(indices, 0)
            idx = indices[0]
            continue
        s = np.delete(sol, N * 2)
        if s[idx] == 0:
            delta = capacidades[idx]
            s[idx] = capacidades[idx]
            diff = True
        else:
            delta = s[idx]
            s[idx] = 0
            diff = False
        balance(s, capacidades, delta, diff)
        sol = np.append(s, sol[N * 2])
        break
    return sol


def anneal(t, t_min, e_th, alpha, N, data, capacidades, demanda):
    f = objective_func
    s = gen_s0(N * 2, capacidades, demanda)
    e = f(s, N, data)
    k = 0
    historial_f = []
    while t > t_min and e > e_th:
        s_new = gen_s(s, N, capacidades, demanda)
        e_new = f(s_new, N, data)
        historial_f.append(e)
        delta = e_new - e
        if delta < 0:
            s = s_new
            e = e_new
        else:
            p = np.exp(-delta / t)
            r = np.random.rand()
            if r < p:
                s = s_new
                e = e_new
        t *= alpha
        k += 1
    return s, k, historial_f


def sa(t_max, t_min, e_th, alpha, N, data, capacidades, demanda):
    x, count, historial_f = anneal(
        t_max, t_min, e_th, alpha, N, data, capacidades, demanda
    )
    return x, historial_f, count


def run_experiment(
    t_inicial, alpha, block_name, N, data, capacidades, demanda, runs=100
):
    def single_run(run, t_inicial, alpha):
        t_start = timeit.default_timer()
        annealing, historial, iterations = sa(
            t_inicial, 1e-5, 1.5e5, alpha, N, data, capacidades, demanda
        )
        t_end = timeit.default_timer()
        costo = objective_func(annealing, N, data)
        return {
            "block": block_name,
            "T_inicial": t_inicial,
            "alpha": alpha,
            "run": run + 1,
            "costo": costo,
            "tiempo": t_end - t_start,
            "iteraciones": iterations,
        }

    results = Parallel(n_jobs=6)(
        delayed(single_run)(i, t_inicial, alpha) for i in range(runs)
    )
    return results


def summarize_experiment(results):
    df = pd.DataFrame(results)
    summary = (
        df.groupby(["block", "T_inicial", "alpha"])
        .agg(
            mean=("costo", "mean"),
            std=("costo", "std"),
            avg_time=("tiempo", "mean"),
        )
        .reset_index()
    )
    summary["var.coeff"] = summary["std"] / summary["mean"]
    summary = summary[
        [
            "block",
            "T_inicial",
            "alpha",
            "mean",
            "var.coeff",
            "avg_time",
        ]
    ]
    return summary


# Parámetros para los experimentos
block1_t_inicial = [50, 75, 100]
block1_alpha = [0.5, 0.7, 0.9]
block2_t_inicial = [100, 200, 300]
block2_alpha = np.arange(0.900, 1.000, 0.011).tolist()

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
    missing_files = []
    for file in archivos.values():
        file_path = os.path.join(folder, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    if missing_files:
        print(
            f"Error: Archivos no encontrados en {folder}: {missing_files}. Saltando escenario {i + 1}."
        )
        continue

    # Cargar datos para el escenario
    try:
        data = model_data(archivos, demanda, folder=folder)
        N, seed, capacidades = model_vars(data["params_df"])
    except Exception as e:
        print(f"Error al cargar datos en {folder}: {e}. Saltando escenario {i + 1}.")
        continue

    # Iniciar medición de tiempo para el escenario
    start_time = timeit.default_timer()
    total_combinations = len(block1_t_inicial) * len(block1_alpha) + len(
        block2_t_inicial
    ) * len(block2_alpha)
    processed_combinations = 0

    # Ejecutar experimentos
    all_results = []
    for t_inicial in block1_t_inicial:
        for alpha in block1_alpha:
            all_results.extend(
                run_experiment(
                    t_inicial, alpha, "Bloque 1", N, data, capacidades, demanda
                )
            )
            processed_combinations += 1
            elapsed_time = timeit.default_timer() - start_time
            avg_time_per_comb = elapsed_time / processed_combinations
            remaining_combinations = total_combinations - processed_combinations
            remaining_time = avg_time_per_comb * remaining_combinations
            print(
                f"Escenario {i + 1}: Procesadas {processed_combinations}/{total_combinations} combinaciones"
                # Faltan aproximadamente {remaining_time / 60:.1f} minutos para terminar.            )
            )

    for t_inicial in block2_t_inicial:
        for alpha in block2_alpha:
            all_results.extend(
                run_experiment(
                    t_inicial, alpha, "Bloque 2", N, data, capacidades, demanda
                )
            )
            processed_combinations += 1
            elapsed_time = timeit.default_timer() - start_time
            avg_time_per_comb = elapsed_time / processed_combinations
            remaining_combinations = total_combinations - processed_combinations
            remaining_time = avg_time_per_comb * remaining_combinations
            print(
                f"Escenario {i + 1}: Procesadas {processed_combinations}/{total_combinations} combinaciones"
                # Faltan aproximadamente {remaining_time / 60:.1f} minutos para terminar."
            )

    # Guardar resultados
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(resultados_folder, "experiment_results_sa.xlsx")
    try:
        results_df.to_excel(results_file, index=False)
        print(
            f"Resultados exportados a '{results_file}' con columnas: block, T_inicial, alpha, run, costo, tiempo, iteraciones"
        )
    except Exception as e:
        print(f"Error al exportar '{results_file}': {e}")

    summary_df = summarize_experiment(all_results)
    summary_file = os.path.join(resultados_folder, "experiment_summary_sa.xlsx")
    try:
        summary_df.to_excel(summary_file, index=False)
        print(
            f"Resumen exportado a '{summary_file}' con columnas: block, T_inicial, alpha, mean, var.coeff, avg_time, avg_iterations"
        )
    except Exception as e:
        print(f"Error al exportar '{summary_file}': {e}")
