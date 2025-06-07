import numpy as np

# Datos de ejemplo
DEMANDA = 40
capacidades = np.array([4, 8, 20, 20, 12])

X = np.array([[0, 0, 20, 20, 0], [0, 8, 15, 5, 12]])
N = X.shape[1]


# Mutar capacidad de un individuo
def mutar_cap(individuo):
    idx_mut = np.random.randint(individuo.shape)
    # False = Menor
    # True = Mayor
    diff = False

    if individuo[idx_mut] == 0:
        delta = capacidades[idx_mut]
        individuo[idx_mut] = capacidades[idx_mut]
        diff = True

    else:
        delta = individuo[idx_mut]
        individuo[idx_mut] = 0

    return igualar(individuo, delta, diff)


# Función para alterar el vector para que cumpla con la restricción de igualdad
def igualar(individuo, delta, diff):
    # Reducir para igualar a la demanda
    if diff:
        acopios = list(np.nonzero(individuo)[0])

        while delta > 0 and len(acopios) > 0:
            idx = np.random.choice(acopios)
            acopios.remove(idx)

            if delta <= individuo[idx]:
                individuo[idx] -= delta[0]
                delta = 0
            else:
                delta -= individuo[idx]
                individuo[idx] = 0
    # Aumentar para igualar a la demanda
    else:
        acopios = list(np.where(individuo == 0)[0])

        while delta > 0 and len(acopios) > 0:
            idx = np.random.choice(acopios)
            acopios.remove(idx)

            if delta <= capacidades[idx]:
                individuo[idx] = delta[0]
                delta = 0
            else:
                individuo[idx] = capacidades[idx]
                delta -= capacidades[idx]


def imprimir(individuo):
    print(f'Vector: {individuo}, Suma: {np.sum(individuo)}')


print(f'Estado inicial')
print(f'Vector: {X[0]}, Suma: {np.sum(X[0])}')
print(f'Vector: {X[1]}, Suma: {np.sum(X[1])}')

for _ in range(1000):
    print('Mutados')
    for i in range(len(X)):
        mutar_cap(X[i])
        imprimir(X[i])
