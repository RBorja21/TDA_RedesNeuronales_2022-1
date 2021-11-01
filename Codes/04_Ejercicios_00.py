# %%
"""
# Ejercicios de repaso de Python, Numpy y Pandas
"""

# %%
"""
## Python
"""

# %%
"""
**1 -** Escriba una función que genere una lista con los primeros n números primos, siendo n un parámetro de dicha función. En la medida de lo posible evite usar un for loop explícito.
"""

# %%
# Inserte su respuesta 

# %%
"""
**2 -** Dada las siguientes dos listas

`
list1 = ["name", "last_name", "age", "loved_foods"]
list2 = ["Juan", "Ramirez", 32, ["pizza", "hamburger", "soda"]]
`

Genere un diccionario mediante un list comprehension, en los cuales las llaves sean los elementos de la primera lista y los valores los elementos de la segunda.
"""

# %%
# Inserte su respuesta 

# %%
"""
**3 -** Explique en detalle la utilidad de emplear bibliotecas como numpy y pandas.
"""

# %%
# Inserte su respuesta.

# %%
"""
**4 -** Se tienen dos listas de números en donde una es permutación de la otra. Sin embargo cuando la información se envió por internet se perdieron algunos números. ¿Cómo identificar qué números se perdieron? (Considere que los números pueden repetirse y la frecuencia debe ser la misma en ambas listas, el resultado se debe presentar una lista de números faltantes en orden ascendente)

Parámetros:

arr: lista original

brr: lista permutada con datos faltantes
"""

# %%
# Inserte su respuesta.

# %%
"""
# Numpy y Pandas
"""

# %%
"""
**5 -** En este ejercicio deberá implementar una regresión lineal múltiple desde cero. Se busca conocer la capacidad de programar un algoritmo, por lo que el detalle de la parte estadística por el momento no es relevante.

La solución a un problema de regresión múltiple está dada por la siguiente expresión:

$$ \hat{\beta} = (X^{T}X)^{-1} X^{T} y \qquad (1)$$


Y la manera de calcular un nuevo valor para $y$ es
$$ y = X \hat{\beta} \qquad (2)$$
"""

# %%
"""
Siendo $X$ la matriz tal que sus filas corresponden a cada valor de $x_{i}$, $y$ el vector
de elementos $y_{i}$ , $\hat{\beta}$ el vector de coeficientes solución
"""

# %%
"""
Dado lo anterior, implemente la clase `LinearRegression` la cual deberá tener los
métodos `fit` y `predict`


El método `fit` deberá recibir como parámetros a la matriz $X$ y al vector $y$, con estos dos deberá calcular $\hat{\beta}$ de acuerdo a la expresión anterior. Este vector de parámetros deberá guardarse manera interna (ver más abajo la forma en que se imprimen dichos coeficientes) para posteriormente ser usado por el método
`predict`.
"""

# %%
"""
El método `predict` deberá recibir la matriz $X$ y generar las predicciones para dicha matriz. Se espera que la clase desarrollada funcione de la siguiente manera
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
class LinearRegression :
    
    def fit(X,y):
        
        '''
        input: X matriz, y vector
        output: beta vector de coeficientes
        '''
        
        beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),(X))),np.transpose(X)),y)
        
        return beta
    def predict(X):
        
        '''
        input: X Matriz 
        output: nueva_y predicción de la matriz X
        '''
        nueva_y = np.dot(X,beta)
        return nueva_y
    
    

# %%
"""
```python
lr = LinearRegression()
lr.fit(X, y)
y_preds = lr.predict(X)


print(lr.coeffs) # coeffs corresponde al vector de coeficientes beta
```
"""

# %%
"""
Finalmente, para mostrar que todo funciona correctamente y ver que tal se encuentran respecto a su conociemoto en `Pandas` y `Numpy`, hagan la prueba con el conjunto de datos `DataSets\casas.csv`
"""

# %%
"""
Los campos son los siguientes: 
- 1. Precio: Precio en que la casa fue vendida.
- 2. Zona: Categoría del vecindario en el que la casa se encuentra. Las categorías van del 0 (la zona más fea) al 4. Se sabe que los números no representan linealmente el ascenso en precio de las casas.
- 3. No_Baños: Número de baños en la casa 
- 4. No_Cuartos: Número de cuartos en la casa 
- 5. Superficie: Superficie de la casa
"""

# %%
"""
**5.1 -** Obtenga algunos estadísticos básicos (promedios, desviaciones, etc.) de esta base de datos sobre las columnas que considere relevantes. Si puede presentar esta información en un gráfico, mejor.

Hint: Algunos precios de ciertas casas los puede considerar como `valores atípicos` (para aquellos que sean muy altos), si los elimina, ¿qué implicaciones tiene?  

Hint: 
```python
plt.figure(figsize=(10, 8))

data.boxplot('Precio')

plt.show()

```
"""

# %%
#Leo el cvc, creo una copia y creo data2 para los datos estadisticos relevantes (pues la zona no es realmente un número importante)
df = pd.read_csv('../Datasets/casas.csv')
data = df.copy()
data2 = data.drop(labels = "Zona", axis = 1)

# %%
#Datos estadísticos básicos
data2.describe().T

# %%
#En caso de eliminar los datos atípicos, se perdería poca información, pero no debería afectar mucho


# %%
"""
**5.2 -** Al momento de capurar la información en la base de datos, por error humano no se agregaron todos los valores correspondientes, diga cuántos valores hacen falta en cada columna. 

Hint: data.isna().sum()
"""

# %%
#El hint resuelve el problema
print(data.isna().sum())

# %%
"""
**5.3 -** Para aquellas renglones que les hagan falta los valores del `precio`, elimine toda la fila, y para quellas columnas que si tengan el valor del `precio` pero les haga falta algún otro campo, implemente una función/método con pandas que utilicé el valor promedio de dicha columana y lo asocie el valor faltante. 
"""

# %%
#eliminan los renglones con NaN en Precio
data = data.dropna(subset = ["Precio"])

# %%
#Rellena los valores faltantes con la media
data = data.fillna(data.mean())

# %%
#establezco el vector y
y1 = data["Precio"] 

# %%
#defino la matriz X
X = data.drop(labels = "Precio", axis = 1)

# %%
"""
**5.4 -** Utilizando las expresiones (1) y (2) y `DataSets\casas.csv` , implementen un modelo de regresión lineal múltiple para predecir el costo de una casa en la zona 2, de 2 baños y 3 recámaras con $250m^{2}$ de superficie.
"""

# %%
#Transformo las matrices de pandas en arreglos de numpy
X = X.to_numpy()
y1 = y1.to_numpy()

# %%
np.shape(X)

# %%
#Acomodo el vector y1 para que no genere conflictos por ser de la forma (391,)
y1 = y1[:, np.newaxis]

# %%
np.shape(y1)

# %%
#Defino el vector de datos de la casa a la que estimaré su precio
yp = np.array([2,2,3,250])
yp = yp[:, np.newaxis]

# %%
np.shape(yp)

# %%
#Utilizo el metodo creado para la clase LinearRegression para obtener la matriz de coeficientes beta gorro, cambio sus dimensiones para que tengan sentido en la siguiente operacion
coeficientes = LinearRegression.fit(X,y1)
coeficientes= np.reshape(coeficientes,(1,4))

# %%
np.shape(coeficientes)

# %%
prediccion = np.dot(coeficientes,yp)
prediccion

# %%
"""
**5.5 -** De una interpretación de los valores de $\hat{\beta}$
"""

# %%
#es el vector que determina el hiperplano que mejor se ajusta a los valores de la matriz.
