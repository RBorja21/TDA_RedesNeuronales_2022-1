# %%
'''
# Numpy


**Objetivo**: Aprender a crear y manipular matrices utilizando la librería Numpy de Python.

**Conocimientos previos**: Programación básica en Python, concepto de vector y matriz.


Numpy es una librería para realizar cálculo **num**érico en **py**thon. La usaremos principalmente porque nos permite crear y modificar matrices, y hacer operaciones sobre ellas con facilidad.

Para comenzar a usar numpy, debemos importar la librería. En este caso la vamos a importar, y darle un sobrenombre al mismo tiempo, **np**, para que el código sea más legible
'''

# %%
# importamos la librería numpy, y le damos como nombre np dentro del programa
import numpy as np

# %%
'''
Ahora que tenemos la librería, empecemos creando un vector de 5 elementos. 

La manera más común de crear una un vector o matriz ya inicializada es con la función **np.array**, que toma una lista (o lista de listas) como parámetro y devuelve una matriz de numpy.
'''

# %%

lista=[25,12,15,66,12.5]
vector=np.array(lista)
print(vector)
print(type(vector))

# %%
'''
¿Cuál es la diferencia entre *vector* y *lista*? Que *vector*, al ser un arreglo de numpy,  nos permite hacer varias operaciones matemáticas de forma muy simple.
'''

# %%
print("- vector original")
print(vector)

print("- sumarle 1 a cada elemento del vector:")
print(vector+1)
print("- multiplicar por 5 cada elemento del vector:")
print(vector*5)

print("- suma de los elementos:")
print(np.sum(vector))

print("- promedio (media) de los elementos:")
print(np.mean(vector)) # 

print("- el vector sumado a si mismo:")
print(vector+vector)
print("- suma de vectores vector1 y vector2 (mismo tamaño):")
vector2=np.array([11,55,1.2,7.4,-8])
print(vector+vector2)

# %%
'''
### Índices y slices de vectores

Así como con las listas, se utilizan los corchetes (*[ ]*) para acceder a sus elementos, y se pueden tomar slices o rebanadas del arreglo utilizando *:* 
'''

# %%
print(vector[3])
print(vector[1:4])
print(vector[1:])
print(vector[:4])
print(vector[:])

# %%
'''
## Creación de vectores con valor 0 o 1

Es muy común crear un vector con valores 0 o 1. Por ejemplo, cuando se utiliza un vector de contadores, donde cada contador comienza en 0. 

Para ello, utilizamos las funciones `np.zeros` y `np.ones`, respectivamente. Cada una toma como parámetro la cantidad de elementos del vector a crear.
'''

# %%
print("- Vector de ceros:")
vector_ceros=np.zeros(5)
print(vector_ceros)

print("- Vector de unos:")
vector_unos=np.ones(5)
print(vector_unos)


#Combinando este tipo de creaciones con las operaciones aritméticas,
#podemos hacer varias inicializaciones muy rápidamente
# Por ejemplo, para crear un vector cuyos valores iniciales son todos 2.

print("- Vector con todos los elementos con valor 2:")
vector_dos=np.zeros(5)+2
print(vector_dos)

print("- Vector con todos los elementos con valor 2 (otra forma):")
vector_dos_otro=np.ones((5))*3
print(vector_dos_otro)

# %%
'''
### Matrices

Los vectores son arreglos de una sola dimensión. Las matrices son arreglos de dos dimensiones; generalmente a la primera dimensión se la llama la de las *filas*, mientras que a la otra se la llama la de las *columnas*.

Por ende, para crearlas con `np.array`, necesitamos no una lista de valores, sino una lista de valores *por cada fila*, o sea, una *lista de listas*. 

Del mismo modo, para crearlas con `np.zeros` o `np.ones`, vamos a necesitar una **tupla** con **dos** elementos, uno por cada dimensión.
'''

# %%
print("- Matriz creada con una lista de listas:")
lista_de_listas=[ [1  ,-4], 
                  [12 , 3], 
                  [7.2, 5]]
matriz = np.array(lista_de_listas)
print(matriz)


print("- Matriz creada con np.zeros:")
dimensiones=(2,3)
matriz_ceros = np.zeros(dimensiones)
print(matriz_ceros)


print("- Matriz creada con np.ones:")
dimensiones=(3,2)
matriz_unos = np.ones(dimensiones)
print(matriz_unos)

#también podemos usar np.copy para copiar una matriz 
print("- Copia de la matriz creada con np.ones:")
matriz_unos_copia=np.copy(matriz_unos)
print(matriz_unos_copia)

# %%
# Ejercicio
# Crear una matriz de 4x9, que esté inicializada con el valor 0.5
#IMPLEMENTAR - COMIENZO
matriz=0 
#IMPLEMENTAR - FIN

print("La matriz es:")
print(matriz)

# %%
'''
### Accediendo a las matrices

También podemos usar slices para acceder a partes de las matrices. Las matrices tienen dos dimensiones, así que ahora tenemos que usar dos indices o slices para seleccionar partes.
'''

# %%
lista_de_listas=[ [1  ,-4], 
                  [12 , 3], 
                  [7.2, 5]]
a = np.array(lista_de_listas)

print("Elementos individuales")
print(a[0,1])
print(a[2,1])

print("Vector de elementos de la fila 1")
print(a[1,:])

print("Vector de elementos de la columna 0")
print(a[:,0])

print("Submatriz de 2x2 con las primeras dos filas")
print(a[0:2,:])

print("Submatriz de 2x2 con las ultimas dos filas")
print(a[1:3,:])

# %%
'''
### Modificando matrices

También podemos usar los slices para modificar matrices. La única diferencia es que ahora los usaremos para seleccionar que parte de la matriz vamos a cambiar.
'''

# %%
lista_de_listas=[ [1,-4], 
                  [12,3], 
                  [7, 5.0]]
a = np.array(lista_de_listas)

print("- Matriz original:")
print(a)

print("- Le asignamos el valor 4 a los elementos de la columna 0:")
a[:,0]=4
print(a)


print("- Dividimos por 3 la columna 1:")
a[:,1]=a[:,1]/3.0
print(a)

print("- Multiplicamos por 5 la fila 1:")
a[1,:]=a[1,:]*5
print(a)

print("- Le sumamos 1 a toda la matriz:")
a=a+1
print(a)

# %%
#Ejercicios

lista_de_listas=[ [-44,12], 
                  [12.0,51], 
                  [1300, -5.0]]
a = np.array(lista_de_listas)

print("Matriz original")
print(a)


# Restarle 5 a la fila 2 de la matriz
print("Luego de restarle 5 a la fila 2:")
#IMPLEMENTAR - COMIENZO
print(a)
#IMPLEMENTAR - FIN

# Multiplicar por 2 toda la matriz
print("Luego de multiplicar por 2 toda la matriz:")
#IMPLEMENTAR - COMIENZO
print(a)
#IMPLEMENTAR - FIN

# Dividir por -5 las dos primeras filas de la matriz
print("Luego de dividir por -5 las primeras dos filas de la matriz:")
#IMPLEMENTAR - COMIENZO
print(a)
#IMPLEMENTAR - FIN


#Imprimir la ultima fila de la matriz
print("La última fila de la matriz:")
#IMPLEMENTAR - COMIENZO
ultima_fila=0 
#IMPLEMENTAR - FIN
print(ultima_fila)

# %%
# Más ejercicios

lista_de_listas=[ [-44,12], 
                  [12.0,51], 
                  [1300, -5.0]]
a = np.array(lista_de_listas)

# Calcular la suma y el promedio de los elementos de a utilizando dos fors anidados
suma = 0
promedio= 0
#IMPLEMENTAR - COMIENZO
print("La suma de los elementos de A es:")
print(suma)
print("El promedio de los elementos de A es:")
print(promedio)
#IMPLEMENTAR - FIN

# Imprimir la suma de los elementos de a utilizando np.sum
#IMPLEMENTAR - COMIENZO
#IMPLEMENTAR - FIN

# Imprimir el promedio de los elementos de a utilizando slices y np.mean
#IMPLEMENTAR - COMIENZO
#IMPLEMENTAR - FIN

# %%
'''
## Array Reshape


1.- **np.shape()** se utiliza para saber las dimensiones de un arreglo de numpy. Esto se puede interpretar como las dimensiones de una matriz.
    
2.- **np.reshape()** se utiliza para cambiar las dimensiones del arreglo (matriz) de numpy.

Estas funciones son muy frecuentes al implementar redes neuronales convolucionales. Esto ya que se suele modelar a las imágenes como un arreglo de pixeles a los cuales les corresponde un valor entre 0 y 255 para cada uno de los colores rojo, verde y azul (RGB). Formando un arreglo de 3 dimensiones ($largo$, $ancho$, $profundo$).

neuronal. En inglés a este vector se le denomina 'flatten'.


Implementemos la función **np.reshape** reordenando un arreglo de la forma (3,3,2) en un arreglo de la forma (18,1):
'''

# %%
def image2vector(image):
    """
    Input:
    imagen -- arreglo de numpy de la forma (longitud, anchura, profundidad)
    
    Returns:
    v -- vector de la forma (longitud*anchura*profundidad, 1)
    """
    
    v = image.reshape( (image.shape[0]*image.shape[1]*image.shape[2]), 1)
    
    return v

# %%
'''
Puedes aprender más del atributo **np.reshape()** en el siguiente [enlace](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html).
'''

# %%
#Definimos un arreglo de prueba y comprobamos:

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

image.shape

# %%
image_flatten = image2vector(image)
image_flatten.shape

# %%

image_flatten

# %%
'''
## Broadcast y normalización
'''

# %%
'''

Un proceso fundamental en algoritmos de Deep learning es la normalización de los datos con lo que se entrenará una red neuronal.
Por ejemplo, es posible normalizar una matriz o arreglo a partir de normalizar cada uno de sus renglones o filas.
Esto es, dada la matriz $$x:
\begin{bmatrix}
    0 & 3 & 4 \\
    2 & 6 & 4 \\
\end{bmatrix}\tag{1}$$ entonces $$\| x\| = \begin{bmatrix}
    5 \\
    \sqrt{56} \\
\end{bmatrix}\tag{2} $$ y        $$ x\_normalized = \frac{x}{\| x\|} = \begin{bmatrix}
    0 & \frac{3}{5} & \frac{4}{5} \\
    \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
\end{bmatrix}\tag{3}$$
'''

# %%
'''
Implementemos una función que nos permita llevar acabo la operación, haremos uso del atributo de numpy **np.linalg.norm(x)**.
'''

# %%
#Función que normaliza cada renglón de un arreglo:

def normalizeRows(x):
    """
    Input:
    x -- Un arreglo de numpy de la forma (n, m)
    
    Output:
    x -- El arreglo x con cada uno de sus renglones normalizado.
    """
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    
    
    # Broadcast
    x = x / x_norm

    return x

# %%
#Probamos la función:

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])

# %%
x_normalizada = normalizeRows(x)
x_normalizada

# %%
'''
Observemos con detenimento la operación $x/x_{norm}$.

Cada uno de estos arreglos tiene las siguientes dimensiones.
'''
print("El arreglo x tiene dimensión {}, mientras que el arreglo x_norm tiene dimensión {}.\n\nPero el arreglo x/x_norm tiene dimensiones {}. Es decir las mismas que x." \
      .format(x.shape,x_norm.shape, x_rowunit.shape)) 

# %%
'''
matriz $\qquad$ $\pm$ $\qquad$  (1, n) $\qquad$  -> $\qquad$  (m, n)

(m, n) $\qquad$  $\frac{*}{÷}$ $\qquad$  (m, 1) $\qquad$  -> $\qquad$  (m, n)
'''      

# %%
'''
##  Sugerencia: No usar estructuras de dato donde la forma sea (n, ) ~ matriz de rango 1. 

**Una estructura de rango 1 no se comporta ni como vector fila ni como vector columna** 

En su lugar (n ,1) o (1, n) según se necesite. Esto para evitar Broadcasting mal ejecutados!!! 
'''

# %%
f = np.array([1, 2, 3, 4])
print(f.shape)
print(f)

# %%
f_buena = f[:, np.newaxis]
print(f_buena.shape)
print(f_buena)

# %%
#Definimos el vector x más la adición de un escalar:

x_escalar = x + 5

print("El arreglo x tiene dimensión {}, mientras que el arreglo x_escalar tiene dimensión {}".format(x.shape,x_escalar.shape)) 

# %% 
'''
Observemos quién es **x_escalar**

## ¿Qué pueden argumentar? 
'''

# %%
'''
Puedes entender mejor el Broadcasting [aquí](https://numpy.org/doc/stable/user/basics.broadcasting.html)

***
'''

# %%
'''
Como un ejercicio extra de broadcasting implementen la función **Softmax**, esta se define como:
    
    
- $ \text{Sea } x \in \mathbb{R}^{1\times n} \text{,       } softmax(x) = softmax(\begin{bmatrix}
    x_1  &&
    x_2 &&
    ...  &&
    x_n  
\end{bmatrix}) = \begin{bmatrix}
     \frac{e^{x_1}}{\sum_{j}e^{x_j}}  &&
    \frac{e^{x_2}}{\sum_{j}e^{x_j}}  &&
    ...  &&
    \frac{e^{x_n}}{\sum_{j}e^{x_j}} 
\end{bmatrix} $ 

- $\text{Para un arreglo } x \in \mathbb{R}^{m \times n} \text{,  $x_{ij}$ asigna al elemento en el $i^{th}$ renglon y $j^{th}$ columna de $x$, entonces tenemos: }$  $$softmax(x) = softmax\begin{bmatrix}
    x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
    x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{m1} & x_{m2} & x_{m3} & \dots  & x_{mn}
\end{bmatrix} = \begin{bmatrix}
    \frac{e^{x_{11}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{12}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{13}}}{\sum_{j}e^{x_{1j}}} & \dots  & \frac{e^{x_{1n}}}{\sum_{j}e^{x_{1j}}} \\
    \frac{e^{x_{21}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{22}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{23}}}{\sum_{j}e^{x_{2j}}} & \dots  & \frac{e^{x_{2n}}}{\sum_{j}e^{x_{2j}}} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \frac{e^{x_{m1}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m2}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m3}}}{\sum_{j}e^{x_{mj}}} & \dots  & \frac{e^{x_{mn}}}{\sum_{j}e^{x_{mj}}}
\end{bmatrix} = \begin{pmatrix}
    softmax\text{(first row of x)}  \\
    softmax\text{(second row of x)} \\
    ...  \\
    softmax\text{(last row of x)} \\
\end{pmatrix} $$
'''

# %%
# EJERCICIO: Implementa la función softmax descrita previamente


#Función Softmax:

def softmax(x):
    """
    Input:
    x -- Un arreglo con dimensiones (n,m)

    Output:
    s -- Un arreglo igual a la función softmax de x con dimensiones (n,m)
    """
    
   #Broadcast
    
    return s

# %%
#Calculamos la función softmax de un arreglo:

x = np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0 ,0]])

# %%
'''
x es un arreglo de dimensiones (2, 5) y softmax(x) es un arreglo con dimensiones (2, 5). 

Por su parte softmax(x) tiene los valores: 

[[9.80897665e-01 8.94462891e-04 1.79657674e-02 1.21052389e-04
  1.21052389e-04]
  
 [8.78679856e-01 1.18916387e-01 8.01252314e-04 8.01252314e-04
  8.01252314e-04]]
'''

# %%
'''
## Vectorización

VECTORIZACIÓN ~ técnicas que perminten "quitar"  bucles explícitos en el código 
'''

# %%
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]



#Implementación del producto punto clásico entre vectores


dot = 0
for i in range(len(x1)):
    dot += x1[i]*x2[i]

dot

# %%
'''
Comparemos con las mismas operaciones utilizando la vectorización de Numpy:
'''

# %%
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

#Producto punto entre vectores de numpy

dot = np.dot(x1,x2)
dot

# %%
'''
## Derivadas numéricas

### Derivada *derecha*

Como bien sabemos del curso de cálculo, la derivada se define como:

$$
f^\prime(x0) = \frac{{\rm d}f}{{\rm d}x}(x_0) \equiv \lim_{h\to 0}
\frac{f(x_0+h)-f(x_0)}{h}.
$$

Numéricamente, es difícil implementar el límite. Olvidándolo por el momento,
el lado derecho de la definición es relativamente sencillo de implementar
numéricamente. Esencialmente requerimos evaluar $f(x)$ en $x_0$ y en $x_0+h$,
donde $h$ es un número (de punto flotante) pequeño. La sutileza está entonces
en implementar por el límite $h\to 0$.
'''

# %%
'''
#### Ejercicio 

- Definan una función `derivada_derecha` que calcule *numéricamente* la
derivada de la función $f(x)$, de una variable (a priori arbitaria), en
un punto $x_0$. Para esto, utilizaremos la aproximación de la derivada
que se basa en su definición, *omitiendo* el límite. Esta función entonces
dependerá de `f`, la función que queremos derivar, `x0` el punto donde queremos
derivar la función, y `h`, que es el incremento *finito* respecto a $x_0$.
Es decir, calcularemos la derivada usando la aproximación
$$
f'(x_0) \approx \frac{\Delta f_+}{\Delta x} \equiv \frac{f(x_0+h)-f(x_0)}{h},
$$
Este método se conoce por el nombre de *diferencias finitas*.

- A fin de simular el $\lim_{h\to 0}$, consideren distintos valores de $h$
cada vez más próximos a cero. Para cada valor de $h$ calculen el error
absoluto del cálculo numérico, es decir, la diferencia del valor calculado
respecto al valor *exacto*. Ilustren esto con una gráfica del error,
para $f(x) = 3x^3-2$, en $x_0=1$. ¿Cuál es el valor de `h` (aproximadamente)
donde obtienen el menor error del cálculo?
'''