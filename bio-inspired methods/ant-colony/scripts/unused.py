print('a')
' Implementación utilizando métodos predefinidos de pymoo como parámetros '

# from pymoo.operators.sampling.rnd import IntegerRandomSampling
# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PM
# from pymoo.operators.repair.rounding import RoundingRepair
#
# algorithm = GA(
#     # pop_size=200,
#     sampling=IntegerRandomSampling(),
#     crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
#     mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
#     eliminate_duplicates=True
# )
#
# res = minimize(model,
#                algorithm,
#                seed=1,
#                verbose=False)
#
# print(f'F: {res.F}, \nX: {res.X}')

' Última implementación de scipy '
# c = tuple(idx for (_, idx) in bounds)
# Aeq = np.ones((1, len(c)))
# beq = 1

# from scipy.optimize import linprog
# from scipy.optimize import differential_evolution

# result = linprog(
#     c,
#     A_ub=Aeq,
#     b_ub=beq,
#     bounds=bounds,
#     method='highs',
#     integrality=1
# )
# print(result)

' Iteraciones para optimizar con pymoo, con cada centro de acopio como el principal'
# for iteration in range(N):
#     model.ca_principal = iteration
#
#     res = minimize(model,
#                    algorithm,
#                    seed=1,
#                    verbose=False)
#
#     print(f'===========> Centro de acopio principal: {params_df['Id_CA'].iloc[iteration]}')
#     # print(f'F: {res.F}, \nX: {res.X}')
#     print_acopios(res.X)