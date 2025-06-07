# cheese
This project aims to solve an optimization problem for a cheese supply chain using different optimizations.
The main methods used are bio-inspired optimizations, done by using already implemented libraries.

# Model
The initially proposed model aims to simplify an already established optimization model *(not shown here)* in the framework of this project.
The variables defined for this model are the following sets:

| **Variable**  | **Description**        |
|---------------|------------------------|
| $`cacopios`$  | Collection centers set |
| $`clientes`$  | Clients set            |
| $`productos`$ | Products set           |

Every collection center has a defined inventory `capacity`, 
this capacity $` stock `$ represents the current cheese capacity of that center given on a consistent unit.
The stock has a defined **cost per unit**, $` price `$.
Each collection center also has a set transport cost to that client,  

## Schema
The $` n `$ collection centers are represented as a vector, each with a defined quantity $` x `$.
The quantity for the $`i`$th position represents the amount of the total demand $` D `$ that collection center $` j_i `$ is going to supply.

$$
\begin{array} {|r|r|r|r|r|r|}
    \hline x_0 & x_1 & x_2 & x_3 & \cdots & x_n \\
    \hline
\end{array}
\quad \therefore \quad x_i = cacopios_i
$$

## Objective function
Minimize the total costs for the supply chain

$$
\begin{align*}
Min(f(x)) &= \sum_{i=0}^{n} \left(cacopios_i \times price_i \right) + transport_i 
\end{align*}
$$

where

$$
\begin{align*}
\sum_{i=0}^{n} x_i &= D \\
x_i &\leq stock_i
\end{align*}
$$

This function operates within the restrictions above

# New model 

Proposed model didn't represent well enough the real optimization problem, 
also it was deemed as *too simple* to use optimization algorithms, so a new model is proposed.

## Variables (wip)

| Variable                | Description |
|-------------------------|-------------|
| $`N`$                   |             |
| $`CA_i`$                |             |
| $`CA_p`$                |             |
| $`k(CA_i)`$             |             |
| $`Precio(CA_i)`$        |             |
| $`cTransp(CA_i)`$       |             |
| $`TiempoAlistam(CA_i)`$ |             |
| $`TiempoMaxDefinido`$   |             |
| $`TiempoTransp(CA_i)`$  |             |
| $`Tiempo(CA_i)`$        |             |
| $`cTiempo`$             |             |
| $`Demanda`$             |             |
| $`Stock(CA_i)`$         |             |
| $`Ppotencial(CA_i)`$    |             |

## Optimization model (wip)
The amount of product to be delivered from every collection center to supply a demand `Demanda`
& which is the best collection center to respond to that demand.

$$
\begin{align*}
    Min(f) = \sum_{i=0 \quad i\neq p}^{N} &\big[ k(CA_i) \times Precio(CA_i) + cTransp(CA_i) + Tiempo(CA_i) \times cTiempo \big] \, + \\
    &\big[ k(CA_p) \times Precio(CA_p) + cTransp(CA_p) + Tiempo(CA_p) \times cTiempo \big]
\end{align*}
$$

Bound to the following restrictions:

$$
\begin{align*}
    \sum_{i=0}^{N} kCA_i &= Demanda \\
    kCA_i &\leq Stock(CA_i) + Ppotencial(CA_i) &\therefore \, i=0,\cdots ,N \\
    TiempoAlistam(CA_i) &\leq TiempoMaxDefinido &\therefore \, i=0,\cdots ,N \\
    Tiempo(CA_i) &= TiempoAlistam(CA_i) + TiempoTransp(CA_i) \\
\end{align*}
$$
