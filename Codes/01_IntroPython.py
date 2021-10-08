# # Tutorial de Python con Jupyter Notebook 

# ## Python
# **Python es un lenguaje de alto nivel, multiparadigma y con tipado dinámico.** 
# Este tutorial no asume conocimiento de Python, pero tampoco explica el lenguaje en detalle. 
# El texto canónico de referencia para cualquier cosa (duda o discusión) relacionada con Python es la propia [documentación oficial de Python](https://docs.python.org/3/).

# ## Jupyter Notebook

# La forma tradicional de ejecutar un programa en python es con el comando `python nombre.py`, donde `nombre.py` es un archivo
# con código fuente python. 

# En lugar de eso, utilizaremos un servidor de Jupyter Notebook con cuadernos de código.  Estos cuadernos (notebooks) nos permiten
# combinar texto y código, organizados en celdas, lo cual es más cómodo para probar cosas nuevas y documentar lo que hacemos. 

# # Python muuy básico 
# Las **variables** en python no necesitan ser declaradas, simplemente se definen al ser utilizadas por primera vez. Además, 
# (si bien no es recomendable) pueden cambiar de tipo volviendo a definir 

x = "Hola Mundo"
print(x)
print(type(x))

x = 5
print(x)
print(type(x))

y = x + 4.3
print(y)
print(type(y))

# ## Tipos de datos básicos 
# ### Números 


r = 3 # Variable de tipo entero 
r 

f = 4.4 #Variable de tipo flotante
f 

# ### Booleanos
# Python implementa todos los operadores usuales de la lógica booleana, usando palabras en inglés (`and, or, not`) en lugar de símbolos (||, &&, !, etc)
# También tiene los típicos operadores de comparación: `<,>,>=,<=,==,!=`

v1 = True #el valor verdadero se escribe True 
v2 = False #el valor verdadero se escribe False
print("- Valores de v1 y v2:")
print(v1,v2)

print("- v1 and v2:")
print(v1 and v2) # y lógico; imprime False
print(v1 or v2)  # o lógico; imprime True
print(not v1)   # negación lógica, imprime False

print(3 == 5)  # Imprime False ya que son distintos
print(3 != 5)  # Imprime True ya que son distintos
print(3 < 5)  # Imprime True ya que 3 es menor que 5

# ### Listas
# Python tiene soporte para listas como un tipo predefinido del lenguaje. 
# Para crear una lista basta con poner cosas entre `[]` (corchetes) y separarlas con `,` (comas).


print("- Lista con 4 números:")
a=[57,45,7,13] # una lista con cuatro números
print(a)

print("- Lista con 3 strings:")
b=["hola",7.7 ,"buen día"] # una lista con tres strings
print(b)

# la función `len` me da la longitud de la lista
print("- Longitud de la lista:")
n=len(a)
print(n)

# Para acceder a sus elementos, se utiliza el []
# Los índices comienzan en 0
print("- Elemento con índice 0 de la lista:")
print(b[0])
print("- Elemento con índice 1 de la lista:")
print(b[1])
print("- Elemento con índice 2 de la lista:")
print(b[2])

print(b)
b.append('Nuevo')
print(b)

una_lista = [1 , 3.4, "cadena"] 
una_lista_de_listas = [[1 , 3.4, "cadena", [1 , 3.4, "cadena"]] ]

# **TODO en python es un objeto** 

# ### Tuplas
# Las tuplas son estructuras de datos imnutables.  Se crean con `()` (paréntesis) en lugar de `[]` (corchetes).


a=(1,2,57,4)
print("- Una tupla de cuatro elementos:")
print(a)
print("- El elemento con índice 2:")
print(a[2])
print("- Los elementos entre los índices 0 y 2:")
print(a[0:2])

# la siguiente línea genera un error de ejecución
a.append(28)

# ### Diccionarios
# Son mutables 
dic = {"d1":3,"d2":4,"d3":"Hey"}

dic2 = {4:"JEJE",2:4}

dic3 = {4:una_lista,3:una_lista_de_listas}

dic3[4][2]

type(dic3)

dic3[4]


# ### Conjuntos
A = set([2,2,3,3,3,4,"Hola"])
A

B = set([3,4,"Hola",5])

# Intersección
B&A

# Diferencia
B-A

# Unión 
A.union(B)
A|B

# ### Estructuras de control

# ### Ciclos for 
for i in range(4):
    #print(i)
    print(una_lista[i])
     
print("Fin")

# For anidado 
for i in una_lista_de_listas:
    for j in i:
        print(j)
        print("Hola")
    
    print("Adios")


# ###  ciclos while
# Son útiles para cuando quieres evaluar condiciones o conjuntos que no están definidos a priori
i=0
while i < len(una_lista):
    print(una_lista[i])
    i=i+1

# # For anidado 
i=0
while i < len(una_lista_de_listas):
    elemento = una_lista_de_listas[i]
    i=i+1
    for j in elemento:
        print(j)
        print("Hola")
    
    print("Adios")   

# ### Booleanos, if, else
a = 200
b = 33
if b > a:
    print("b es mayor que a")
elif a == b:
    print("a y b son iguales")
else:
    print("a es mayor que  b")

# Romper ciclos 


i=0
while i < len(una_lista_de_listas):
    if i ==1:
        break
    elemento = una_lista_de_listas[i]
    i=i+1
   
        
    for j in elemento:
        print(j)
        print("Hola")
    
    print("Adios")


# ##  Funciones

def nombre_funcion(argumentos):
    operaciones 
    return resultado 


def calculadora(num1,num2,operacion):
    if operacion == "suma":
        result = num1 + num2    
    elif operacion == "resta":
        result = num1 - num2
    elif operacion == "producto":
        result = num1*num2
    elif operacion == "cociente":
        result = num1/num2
    else:
        result = "Operación desconocida, operaciones válidas: suma, resta, producto, cociente"
    return result

calculadora(4,5,"suma")


def cuadrado(x):
    return x**2

cuadrado(3)

ejemplo = lambda x: x**2 
ejemplo(3)

# **Ejercicio**
# Escribir una función que reciba una lista y un valor, 
# y devuelva la cantidad de veces que aparece ese valor en la lista

def ocurrencias(lista,valor):
    # IMPLEMENTAR
    return 0


l=[1,4,2,3,5,1,4,2,3,6,1,7,1,3,5,1,1,5,3,2]
v=2

print("La cantidad de ocurrencias es:")
print(ocurrencias(l,v))
# debe imprimir 3, la cantidad de veces que aparece el 2 en la lista
