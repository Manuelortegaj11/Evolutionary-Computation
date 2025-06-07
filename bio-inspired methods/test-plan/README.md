# Plan de ejecución para pruebas de los métodos bioinspirados

Se definen 2 bloques de parámetros para probar con cada algoritmo
1. Exploratorio
2. Balanceado (debe tener mejores resultados que el bloque 1)

Para todos los algoritmos se debe recopilar los siguientes criterios de optimización

#### Optimalidad
- Función objetivo
- Gráfica de convergencia (tomar la mejor configuración de los bloques, y hacer una ejecución para graficar)
    - La mejor configuración se toma con la función objetivo y el coeficiente de variación más bajos

#### Variabilidad del método
- Media
- Coeficiente de variación: $`\frac{\sigma}{\bar{X}}`$

#### Tiempo computacional
- Individual (por iteración)
- Promedio (múltiples pruebas)

## SA

### Definir parámetros

#### Bloque 1

1. Temperatura inicial
   $` T_{inicial} = [50, 75, 100] `$
2. Tasa de enfriamiento (alpha)
   $` \alpha = [0.5, 0.7, 0.9] `$
3. Temperatura final
   $` T_{final} = 1e-5 `$
4. Máximo de iteraciones
    -  De acuerdo a los parámetros de arriba, evitar que el método se detenga por ese criterio

#### Bloque 2

1. Temperatura inicial
   $` T_{inicial} = [100, 200, 300] `$
2. Tasa de enfriamiento (alpha)
   $` \alpha = \left\{ 0.900 \rightarrow 0.999 : \alpha_{i+1} = \alpha_i + 0.011 \right\} `$
3. Máximo de iteraciones
    -  De acuerdo a los parámetros de arriba, evitar que el método se detenga por ese criterio

## GA

### Definir parámetros

#### Bloque 1

1. Tasa de mutación
   $` T_{mutación} = [0.5, 0.7, 0.9] `$
2. Tasa de cruce
   $` T_{cruce} = [0.5, 0.7, 0.9] `$
3. Tamaño de la población
   $` size_{pob} = [50, 100, 150] `$
4. Número de generaciones
   $` N_{gen} = [100, 300, 500] `$

#### Bloque 2

1. Tasa de mutación
   $` T_{mutación} = \left\{ 0.1 \rightarrow 0.5 : T_{mutación+1} = T_{mutación} + 0.1 \right\} `$
2. Tasa de cruce
   $` T_{cruce} = \left\{ 0.1 \rightarrow 0.7 : T_{cruce+1} = T_{cruce} + 0.1 \right\} `$

## ACO

### Definir parámetros

#### Bloque 1

1. Tasa evaporación
   $` T_{evaporación} = [0.5, 0.7, 0.9] `$
2. Tamaño colonia
   $` size_{col} = [20, 30, 50] `$
3. Número de generaciones
   $` N_{gen} = [50, 100, 150] `$
4. alpha
   $` \alpha = [1.0, 1.5, 2.0] `$
5. beta
   $` \beta = [1.0, 1.5, 2.0] `$

#### Bloque 2

1. Tasa evaporación
   $` T_{evaporación} = \left\{ 0.900 \rightarrow 0.999 : T_{evaporación_{i+1}} = T_{evaporación_i} + 0.011 \right\} `$