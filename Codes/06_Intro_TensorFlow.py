# %%
"""
# Introducción a TensorFlow
"""

# %%
"""
<img src="../Imagenes/tf.png" style="width:300px;height:250px;">
"""

# %%
"""
Pensamos a TensorFlow como una biblioteca de programación numérica de alto rendimiento para el flujo de tensores. 

<img src="../Imagenes/SS-58.png" style="width:500px;height:500px;">

"""

# %%
"""
<img src="../Imagenes/Ejemplo02.png" style="width:300px;height:300px;">
"""

# %%
"""
¿Qué significa que los datos fluyan?

Un **tensor** es una matriz de datos n-dimensional

- Rank 0, Tensor scalar ~ el dato más simple que se puede tener.
- Rank 1, Tensor vector 
- Rank 2, Tensor matrix 
- Rank 3, Tensor
- Rank 4, Tensor 
"""

# %%
"""
<img src="../Imagenes/SS-59.png" style="width:400px;height:400px;">
"""

# %%
"""
Por tanto, un **tensor** es una matriz de datos n-dimensional. Son estos tensores que fluyen a través grafo,
de ahí el nombre de TensorFlow. 
"""

# %%
"""
**TensorFlow contiene varias capas de abstracción** 

- La capa más baja de abstracción es la capa que se implementa para apuntar a las diferentes plataformas de hardware. 

<img src="../Imagenes/SS-61.png" style="width:750px;height:400px;">
"""

# %%
"""
## Componentes de TensorFlow: Tensors and Variables
"""

# %%
"""

- Un tensor es una matriz de datos N-dimensional

Se comportan como matrices n-dimensionales numpy *excepto* que:

  - **tf.constant** produce tensores constantes
  - **tf.Variable** produce tensores que pueden modificarse
"""

# %%
"""
<img src="../Imagenes/SS-62.png" style="width:750px;height:460px;">
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.animation as animation  



import tensorflow as tf

writergif = animation.PillowWriter(fps=30)

# %%
x = tf.constant([
    [3, 5, 7],
    [4, 6, 8]])


y = x[:, 1]
y

# %%
y.numpy()

# %%
z = tf.reshape(x, [6, 1])
z

# %%
# x <- 2 
x = tf.Variable(2.0, dtype=tf.float32, name='my_variable')

# %%
x

# %%
# x <- x + 4 
x.assign_add(4)

# %%
# w * x
w = tf.Variable([[1.], [2.]])
x = tf.constant([[3., 4]])
tf.matmul(w, x)

# %%
"""
TensorFlow tiene la capacidad de calcular la derivada parcial de cualquier función con respecto a cualquier variable. Sabemos que durante el entramiento, los pesos se actualizan utilizando la derivada parcial de la función de perdida con respecto a cada individuo. 
"""

# %%
"""
#### GradientTape registra operaciones para diferenciación automática.

**TensorFlow puede calcular la derivada de una función con respecto a cualquier parámetro.**


- el cálculo se registra con **GradientTape**

"""

# %%
def compute_gradients(X, Y, w0, w1): 
    with tf.GradientTape() as tape: # <- registra el cálculo con GradientTape
        loss = loss_mse(X, Y, w0, w1)#   cuando se ejecuta (¡no cuando está definido!)
        
    return tape.gradient(loss, [w0, w1]) #<- Especifique la función (pérdida) así como los 
                                         # parámetros con los que desea tomar el gradiente 
                                         # respecto a ([w0, w1])

# %%
w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

# %%
dw0, dw1 = compute_gradients(X, Y, w0, w1) #Ejemplo solamente, falta definir los valores de X, Y

# %%
"""
### Cálculo de gradientes usando `tf.GradientTape ()` 
"""

# %%
x = tf.Variable(5.0)

with tf.GradientTape() as tape:
    y = x**2 

# %%
y # ~ y=x**2 = 5**2 

# %%
dy_dx = tape.gradient(y, x) #calcula el gradiente
dy_dx 

# %%
"""
Otro ejemplo
"""

# %%
W = tf.Variable(tf.random.normal((4,2)))
W

# %%
b = tf.Variable(tf.ones(2, dtype=tf.float32))
b

# %%
x = tf.Variable([[10., 20., 30., 40.]], dtype=tf.float32)
x

# %%
"""
Gradient Tape resources

Se utilizan tan pronto como se llama a tape.gradient ()
"""

# %%
with tf.GradientTape(persistent=True) as tape: #To compute multiple gradients over
                                                #the same computation, create a persistent gradient tape
    y = tf.matmul(x, W) + b
    
    loss = tf.reduce_mean(y**2)

# %%
#Habrá registrado todas las operaciones que se realizaron 

#Cálcula el gradiente de loss con respecto a las variable W y b

[d1_dw, d1_db] = tape.gradient(loss, [W, b])

# %%
d1_dw #preserva las dimensiones (shape)

# %%
d1_db

# %%
layer = tf.keras.layers.Dense(2, activation='relu')
x = tf.constant([[10., 20., 20.]])

# %%
# las capas de Keras pueden invocarse como funciones 
with tf.GradientTape() as tape:
    y = layer(x)
    
    loss = tf.reduce_mean(y**2)
    
# calcula el gradiente respecto a todos los parámetros entrenables de la red    
grad = tape.gradient(loss, layer.trainable_variables)

# %%
grad

# %%
"""
GradientTape observa variables entrenables

Los tensores, constantes y variables no entrenables no se rastrean automáticamente
"""

# %%
x1 = tf.Variable(5.0) #entrenable por defecto
x1

# %%
x2 = tf.Variable(5.0, trainable=False) #No entrenable 
x2

# %%
x3 = tf.add(x1, x2)
#x3 = tf.Variable(6.0)
x3

# %%
with tf.GradientTape() as tape:
    
    y = (x1**2) + (x2**2) + (x3**2) 
    
grad = tape.gradient(y, [x1, x2, x3])
grad

# %%
"""
## Ejemplo 

# Regresión lineal simple
"""

# %%
W_true = 2
b_true = 0.5

# %%
x = np.linspace(0, 3, 130)
y = W_true * x + b_true + np.random.randn(x.shape[0]) * 0.5

# %%
plt.figure(figsize=(10, 8))
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()

# %%
class LinearModel:
    
    def __init__(self):
        self.weight = tf.Variable(np.random.randn(), name='w')
        self.bias = tf.Variable(np.random.randn(), name='b')
        
    def __call__(self, x):
        return self.weight* x + self.bias

# %%
def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

# %%
def train(linear_model, x, y, lr = 0.01):
    
    with tf.GradientTape() as tape:
        
        y_pred = linear_model(x)
        current_loss = loss(y, y_pred)
    
    d_weight, d_bias = tape.gradient(current_loss,
                                    [linear_model.weight, linear_model.bias])
    
    linear_model.weight.assign_sub(lr * d_weight)
    linear_model.bias.assign_sub(lr * d_bias)

# %%
linear_model = LinearModel()

weights, biases = [], []


# Mueva estos hiperparámetros y vea como afecta al resultado 
epochs = 10
lr = 0.2

# %%
for epoch_count in range(epochs):
    
    weights.append(linear_model.weight.numpy())
    biases.append(linear_model.bias.numpy())
    
    real_loss = loss(y, linear_model(x))
    
    train(linear_model, x, y, lr=lr)
    
    print(f"Epoch count {epoch_count} W: {weights[epoch_count]} bias: {biases[epoch_count]} Loss value: {real_loss.numpy()}")

# %%
plt.figure(figsize=(10, 8))

plt.plot(range(epochs), weights, 'r', range(epochs), biases, 'b')
plt.plot([W_true] * epochs, 'r--', [b_true] * epochs, 'b--')

plt.legend(['W', 'b', 'true W', 'true b'])
plt.show()

# %%
linear_model.weight.numpy(), linear_model.bias.numpy()

# %%
rmse = loss(y, linear_model(x))
rmse.numpy()

# %%
plt.figure(figsize=(10, 8))

plt.plot(x, y, 'ro', label = 'Original data')
plt.plot(x, linear_model(x), label = 'Fitted line')

plt.title('Linear Regression')

plt.legend()
plt.show()

# %%
