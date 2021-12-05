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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import math

# %%
def nprimos(n):
    resultado=[]
    posibleprimo = 2
    k_primos=0
    while(k_primos<n):
        contador_de_divisores=0
        if (posibleprimo == 2):
            resultado.append(2)
            k_primos=k_primos+1
            posibleprimo=posibleprimo+1
            
        else:
            for j in range(1,math.ceil(math.sqrt(posibleprimo))+1):
                if (posibleprimo%j==0):
                    contador_de_divisores=contador_de_divisores+1
            if (contador_de_divisores==1):
                resultado.append(posibleprimo)
                k_primos=k_primos+1
            posibleprimo=posibleprimo+1
    print(resultado)
            

    
nprimos(10000)

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
list1 = ["name", "last_name", "age", "loved_foods"]
list2 = ["Juan", "Ramirez", 32, ["pizza", "hamburger", "soda"]]

respuesta= {x:y for (x,y) in zip(list1,list2)}
print(respuesta)

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
def funcion_respuesta(arr,brr):
    for i in brr:
        arr.remove(i)
        arr.sort()
    return arr
print(funcion_respuesta([1,2,3,1,2,2,3],[3,1,2,1]))

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
    def predict(X,beta):
        
        '''
        input: X Matriz, beta vector de coeficientes
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
X

# %%
"""
**5.4 -** Utilizando las expresiones (1) y (2) y `DataSets\casas.csv` , implementen un modelo de regresión lineal múltiple para predecir el costo de una casa en la zona 2, de 2 baños y 3 recámaras con $250m^{2}$ de superficie.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
#Leo el csv, creo una copia, elimino los renglones sin precio, defino el vector y, obtengo los cuantiles 95 y 05 para eliminar datos atipicos
df = pd.read_csv('../Datasets/casas.csv')
data = df.copy()
data = data.dropna(subset = ["Precio"])
y=data["Precio"]
max_precio = y.quantile(0.95)
min_precio = y.quantile(0.05)

# %%
#elimino datos atípicos
data = data[(y<max_precio)&(y>min_precio)]

# %%
#relleno los datos vacíos y redefino al vector y
data = data.fillna(data.mean())
y = data["Precio"]

# %%
data.head()

# %%
data.columns

# %%
#Defino los que serán proximamente vectores de la matriz X
x1=data["Zona"]
x2=data["No_Baños"]
x3=data["No_Cuartos"]
x4=data["Superficie"]

# %%
print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x4.shape)
print(y.shape)

# %%
#convierto los datos en arreglos de numpy
x1=np.array(x1)
x2=np.array(x2)
x3=np.array(x3)
x4=np.array(x4)
y=np.array(y)

# %%
n=len(x1)
n

# %%
#Creo el vector de unos
x_bias=np.ones((n,1))

# %%
#Arreglo las dimensiones de los xi
x1_new=np.reshape(x1,(n,1))
x2_new=np.reshape(x2,(n,1))
x3_new=np.reshape(x3,(n,1))
x4_new=np.reshape(x4,(n,1))

# %%
#Creo la matriz X añadiendo las columnas que corresponden a cada xi
x_new=np.append(x_bias,x1_new,axis=1)
x_new=np.append(x_new,x2_new,axis=1)
x_new=np.append(x_new,x3_new,axis=1)
x_new=np.append(x_new,x4_new,axis=1)

# %%
x_new

# %%
x_new_transpose=np.transpose(x_new)
x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)
temp_1=np.linalg.inv(x_new_transpose_dot_x_new)
temp_2=x_new_transpose.dot(y)

# %%
#obtengo los coeficientes de beta gorro en la variable theta
theta=temp_1.dot(temp_2)
theta

# %%
beta0=theta[0]
beta1=theta[1]
beta2=theta[2]
beta3=theta[3]
beta4=theta[4]
print(beta0)
print(beta1)
print(beta2)
print(beta3)
print(beta4)

# %%
#Hago la predicción del precio de la casa usando los coeficientes
beta0 + beta1*2 + beta2*2 + beta3*3 + beta4*250

# %%
"""
**5.5 -** De una interpretación de los valores de $\hat{\beta}$
"""

# %%
#es el vector que determina el hiperplano que mejor se ajusta a los valores de la matriz.
