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
# Inserte su respuesta.
import pandas as pd
import numpy as np

datos=pd.read_csv('../DataSets/casas.csv')
print(datos.info())
print(datos.head())

nuevo=pd.DataFrame(datos)
print(nuevo)
nuevo=nuevo.replace(np.nan,"0")
print("***Impresión sin NaN***")
print(nuevo.info())
print("\n"*5)
print("***Estadisticas sin NaN***")
print(nuevo.describe())





# %%
"""
**5.2 -** Al momento de capurar la información en la base de datos, por error humano no se agregaron todos los valores correspondientes, diga cuántos valores hacen falta en cada columna. 

Hint: data.isna().sum()
"""

# %%
# Inserte su respuesta.
import pandas as pd
import numpy as np

datos=pd.read_csv('../DataSets/casas.csv')
nuevo=datos[['Precio','Zona','No_Baños','No_Cuartos','Superficie']]
nuevo.isna().sum()



# %%
"""
**5.3 -** Para aquellas renglones que les hagan falta los valores del `precio`, elimine toda la fila, y para quellas columnas que si tengan el valor del `precio` pero les haga falta algún otro campo, implemente una función/método con pandas que utilicé el valor promedio de dicha columana y lo asocie el valor faltante. 
"""

# %%
# Inserte su respuesta.

# %%
"""
**5.4 -** Utilizando las expresiones (1) y (2) y `DataSets\casas.csv` , implementen un modelo de regresión lineal múltiple para predecir el costo de una casa en la zona 2, de 2 baños y 3 recámaras con $250m^{2}$ de superficie.
"""

# %%
# Inserte su respuesta.
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

datos=pd.read_csv('../DataSets/casas.csv')
nuevo=datos[['Precio','Zona','No_Baños','No_Cuartos','Superficie']]

datos=datos.replace(np.nan,"0")
precio=datos['Precio'].values
zona=datos['Zona'].values
baños=datos['No_Baños'].values
cuartos=datos['No_Cuartos'].values
metros=datos['Superficie'].values

X=np.array([zona,baños,cuartos,metros]).T
Y=np.array(precio)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)
error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)

print("El error es: ",error)
print("El valor de r² es: ",r2)
print("Los coeficientes son: ",reg.coef_)
zona=2
baños=2
cuartos=3
metros=250
print("Costo de la predicción: ",reg.predict([[zona,baños,cuartos,metros]]))



# %%
"""
**5.5 -** De una interpretación de los valores de $\hat{\beta}$
"""

# %%
# Inserte su respuesta.
