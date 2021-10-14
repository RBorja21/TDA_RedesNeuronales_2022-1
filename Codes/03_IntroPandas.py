# %%
"""
# Introducción a Pandas
"""

# %%
"""
Primero vamos a entender las estructuras de datos principales de `Pandas`. 
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
"""
## Primera estructura: Las series de Pandas
"""

# %%
"""
La estructura elemental de panda son las *series*. Para declararlas, se puede usar la función con el constructor por defecto `Series`.
"""

# %%
pd.Series(dtype='float64')

# %%
"""
Esta estructura de datos es muy parecida a un `np.array` o a una lista de datos en `Python` con la particularidad que existen índices y funciones específicas para manejar la serie. Podemos construir estas series desde una lista, desde un `np.array` o desde un diccionario:
"""

# %%
#Con una lista
lista = [1, 2, 3, 4, 15.1]
serie_lista=pd.Series(lista)
serie_lista

# %%
#Con un nparray
arreglo = np.array(lista)
serie_arreglo=pd.Series(arreglo)
serie_arreglo

# %%
#Con un diccionario
diccionario = {'a' : 1, 'b' : 2, 'c' : "perro", 'd' : 4, 'e' : 5}
serie_diccionario = pd.Series(diccionario)
serie_diccionario

# %%
diccionario_2 = {'perro':200, 'Cristino':"Ronaldo" }
diccionario_2

# %%
nueva_serie = pd.Series(diccionario_2)
nueva_serie

# %%
lista = [0,15,338,2, ';)']

# %%
serie_1 = pd.Series(data=lista, index=['Hola', 'Juan', 'Carlos','¿Cómo estás?', '...'])

# %%
serie_1

# %%
diccionario = {'Rojo':0,'Azul':15,'Verde':338}

# %%
serie_2 = pd.Series(diccionario)

# %%
serie_2

# %%
"""
En el último caso, el diccionario siempre está compuesto por pares *llave, valor*. En este caso, la llave se va a convertir en un índice, con lo que podemos acceder a los elementos por la misma llave (como un diccionario).
"""

# %%
serie_2[0]

# %%
serie_diccionario["c"]

# %%
"""
Bien podríamos especificar cualquier otro valor para los índices. Cuando usamos el constructor de una variable con de la clase `Series`, se toma el argumento como el parámetro `data`. Es decir, las siguientes dos instrucciones hacen lo mismo:
"""

# %%
pd.Series(data=lista)

# %%
pd.Series(diccionario)

# %%
"""
Bien, pues si queremos especificar índices sin usar un diccionario, se puede usar el parámetro `index`.
"""

# %%
mSerie = pd.Series(data = lista, index = ['24','67','68','100'])
mSerie

# %%
"""
Entre los métodos más comunes para aplicar a una serie están `mean` (media), `quantiles` (para calcular cuantiles, el argumento `[0.25, 0.5, 0.75]` regresa los cuartiles), `std` (calcula la desviación estándar). Para métodos más sofisticados, existe la función `apply`, que permite aplicar una función a cada entrada de la serie (esto va a tener más utilidad después). Con el uso de lambdas, podemos, por ejemplo, meter ruido gaussiano a una serie (hay otras maneras más eficientes de hacer esto, pero es un ejemplo forzado).
"""

# %%
time = np.linspace(0,2*np.pi, 500)
mGauss = pd.Series(np.sin(time))
mGaussR = mGauss.apply(lambda x: float(x + np.random.normal(0,0.1,1)))
plt.plot(time,mGauss)

# %%
plt.plot(time, mGaussR)

# %%
mGaussR

# %%
"""
Podemos hacer un histograma con la opción 'plot', e incluso especificar el número de "columnas".
"""

# %%
(mGaussR - np.sin(time)).plot(kind = "hist", bins = 100)

# %%
"""
## Segunda estructura: Data Frame
"""

# %%
"""
Un Data Frame es una colección de series. Ya está. Eso es todo. Por ejemplo, una creación de un data frame es la siguiente:
"""

# %%
concentracion = pd.Series(np.abs(np.random.normal(0,1,5)))
temperatura = pd.Series(np.random.normal(273,4,5))
componentes = pd.DataFrame({'PrimeraColumna': concentracion, 'SegundaColumna':temperatura})
componentes

# %%
"""
Podemos obtener las series de datos usando el nombre de sus campos, y de esa manera usar las operaciones que ya conocemos sobre una serie:

"""

# %%
componentes["SegundaColumna"]

# %%
"""
Podemos añadir columnas al dataframe sólamente dándoles nombre:
"""

# %%
componentes["Temp.Celsius"] = componentes["SegundaColumna"] - 273.15
#componentes.insert(1,"OtraColumna",componentes["Temp.Celsius"])

# %%
componentes

# %%
"""
Y de la misma manera, quitar la columna.
"""

# %%
componentes2=componentes.drop(labels = "Temp.Celsius", axis = 1)
componentes2

# %%
"""
La función miembro drop es muy útil, pero para ello hay que conocer otras particularidades de Pandas. Una de las características más importantes de un Data frame es que puede manejar información faltante.
"""

# %%
componentes.loc["5","Presión"] = np.abs(np.random.normal(0,1,1))
componentes

# %%
"""
En este caso, el último renglón carece de información en las columnas de "Concentración" y "Presión". Para los primeros 5 renglones, el campo de presión es desconocido. Python nos permite deshacernos de renglones que contengan valores NaN, por ejemplo:
"""

# %%
componentes.dropna()

# %%
componentes.dropna(subset = ["PrimeraColumna"])

# %%
"""
Nótese que así como existe `dropna`, existe `fillna` con lo que podemos reemplazar NaN's facilmente:
"""

# %%
x=15

componentes.fillna(x)

# %%
"""
Note que el método `mean` aplicado a un DataFrame genera el promedio por cada columna. Con ello, podemos usar `fillna` de una manera un tanto más sofisticada.
"""

# %%
componentes.fillna(componentes.mean())

# %%
componentes

# %%
"""
También se puede ir llenando "a mano".
"""

# %%
x = pd.DataFrame(columns = ['Código','Nombre','Precio'])
x.loc[0] = ['0232','Pato a lorang', 125.50]
x.loc[1] = ['0231','Pato a lorang 2', 125.70]
x.loc[2] = ['0237','Pato pro', 155.50]
x.loc[3] = ['0222','Pato superpro', 12.50]
x.loc[4] = ['0212','Pato a la', 15.50]
x

# %%
data = pd.DataFrame(columns=['Text', 'Number'])
data.loc[0] = ['Hello', 12345]
data.loc[1] = ['Caroline', 87645]
data

# %%
otherfunc = lambda row: str(row)+'!!'
data.apply(lambda column: column.apply(otherfunc), axis = 0)

# %%
otherfunc_2 = np.vectorize(lambda row: str(row) + '!!!?')
colNames = data.columns
pd.DataFrame(otherfunc_2(data), columns=colNames)

# %%
