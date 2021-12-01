# %%
"""
# Implementación de una  Red Neuronal Profunda utilizando la Api Secuencial en Keras (Sequential API in Keras)
"""

# %%
"""
## Ejercicio: Predecir la esperanza de vida en un pais por medio de un modelo de regresión
"""

# %%
import os, datetime

import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.metrics import r2_score #métricas
from sklearn.preprocessing import StandardScaler #Estandarizar datos 

import tensorflow as tf 

# %%
data = pd.read_csv('../DataSets/LifeExpectancyData.csv')
data.sample(5)

# %%
"""
<img src="../Imagenes/meme.jpg" style="width:500px;height:450px;">
"""

# %%
data.shape

# %%
#Valores faltantes
data.isna().sum()

# %%
countries = data['Country'].unique()
#data['Life expectancy']

na_cols = ['Life expectancy ', 'Adult Mortality', 'Alcohol', 'Hepatitis B',
          ' BMI', 'Polio', 'Total expenditure', 'Diphtheria ', 'GDP',
          ' thinness  1-19 years', ' thinness 5-9 years','Population',
          'Income composition of resources']

for col in na_cols:
    for country in countries:
        data.loc[data['Country'] == country, col] = data.loc[data['Country'] == country, col]\
                                                        .fillna(data[data['Country'] == country][col].mean())  
                                        #fillna para los valores faltantes
                                        # Fill NA/NaN values using the specified method

# %%
data.isna().sum()

# %%
"""
Todavía faltan varios datos, por lo que es probable que falten todos los valores para esa respectiva columna en particular para un país 
"""

# %%
data = data.dropna() # Elimina los valores faltantes.
data.shape

# %%
data.isna().sum()

# %%
data['Status'].value_counts()

# %%
data['Country'].value_counts() #16 años de registros para cada pais

# %%
plt.figure(figsize=(10, 8))

data.boxplot('Life expectancy ')

plt.show()

# %%
plt.figure(figsize=(8, 6))

sns.boxplot('Status', 'Life expectancy ', data=data)

plt.xlabel('Status', fontsize=16)
plt.ylabel('Life expectancy ', fontsize=16)

plt.show()

# %%
plt.figure(figsize=(8, 6))

sns.boxplot('Status', 'Total expenditure', data = data)

plt.xlabel('Status', fontsize=16)
plt.ylabel('Total expenditure', fontsize=16)

plt.show()

# %%
#Correlaciones 

data_corr = data[['Life expectancy ',
                 'Adult Mortality',
                 'Schooling',
                 'Total expenditure',
                 'Diphtheria ',
                 'GDP',
                 'Population']].corr()

data_corr

# %%
"""
El coeficiente de correlación es una medida de la relación lineal que existe entre las variables, y es un valor entre menos uno a uno. ~ se mueven en la misma dirección (positivo) o contraria (negativo)
"""

# %%
ig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(data_corr, annot=True)

plt.show()

# %%
features = data.drop('Life expectancy ', axis=1) #Elimina las etiquetas especificadas de filas o columnas..

target = data[['Life expectancy ']]

# %%
features.columns

# %%
target.sample(5)

# %%
features = features.drop('Country', axis=1)
features.columns

# %%
features.head(10)

# %%
categorical_features = features['Status'].copy()
categorical_features.head()

# %%
#Convertir variables categóricas en variables 
# one hot encodding
categorical_features = pd.get_dummies(categorical_features)
categorical_features.tail()

# %%
numeric_features = features.drop(['Status'], axis=1)

numeric_features.head()

# %%
numeric_features.describe().T

# %%
"""
Podemos ver que la desviación estandar y la media tiene valores muy diferentes para las features. 

Los modelos de aprendizaje automático, especialmente los modelos de redes neuronales, tienden a ser mucho más robustos cuando se estandarizan los datos de entrada.
"""

# %%
standardScaler = StandardScaler()

numeric_features = pd.DataFrame(standardScaler.fit_transform(numeric_features),
                               columns=numeric_features.columns,
                               index=numeric_features.index)

numeric_features.describe().T

# %%
processed_features = pd.concat([numeric_features, categorical_features], axis=1,
                              sort=False)

processed_features.head()

# %%
processed_features.shape

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(processed_features,
                                                   target, 
                                                   test_size=0.2,
                                                   random_state=1)

# %%
(x_train.shape, x_test.shape), (y_train.shape, y_test.shape)

# %%
y_train

# %%
x_train

# %%
def build_single_layer_model():
    
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(32, #32 neuronas
                                   input_shape = (x_train.shape[1], ), #forma de entrada
                                   activation = 'sigmoid'))
    
    model.add(tf.keras.layers.Dense(1)) #capa de salida 
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    model.compile(loss = 'mse',
                 metrics = ['mae', 'mse'], #mae ~ mean absolute error
                 optimizer = optimizer)
    
    return model

# %%
model = build_single_layer_model()
model.summary()

# %%
tf.keras.utils.plot_model(model)

# %%
num_epochs = 100

training_history = model.fit(x_train,
                            y_train,
                            epochs=num_epochs,
                            validation_split=0.2,# 20% de los datos de entrenamiento se utilizan para validar
                            verbose=True)

# %%
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.plot(training_history.history['mae'])
plt.plot(training_history.history['val_mae'])

plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'val'])

plt.subplot(1, 2, 2)

plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])

# %%
model.evaluate(x_test, y_test)

# %%
y_pred = model.predict(x_test)

#Como es un modelo de regresión una mejor métrica que podemos usar para evaluar nuestro modelo
# es r2 score
r2_score(y_test, y_pred)

# %%
pred_results = pd.DataFrame({'y_test': y_test.values.flatten(),
                             'y_pred': y_pred.flatten()}, index=range(len(y_pred)))
    
pred_results.sample(10)

# %%
plt.figure(figsize=(10, 8))

plt.scatter(y_test, y_pred, s=100, c='blue')

plt.xlabel('Actual life expectancy values')
plt.ylabel('Predicted life expectancy values')

plt.show()

# %%
"""
### Tiene una  relación lineal, lo que indica que los valores reales están muy cerca de los pronosticados!!!
"""

# %%
def build_multiple_layer_model():
    
    model = tf.keras.Sequential([tf.keras.layers.Dense(32, input_shape = (x_train.shape[1], ), activation = 'relu'),
                              tf.keras.layers.Dense(16, activation = 'relu'),
                              tf.keras.layers.Dense(4, activation = 'relu'),
                              tf.keras.layers.Dense(1)])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(loss = 'mse', metrics = ['mae', 'mse'], optimizer = optimizer)
    
    return model

# %%
model = build_multiple_layer_model()
tf.keras.utils.plot_model(model, show_shapes=True)

# %%
"""
El `None` es porque le tamaño del batch es desconocido
"""

# %%
logdir = os.path.join("seq_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq = 1)

# %%
checkpoint_filepath = '../ModelsSaved/Ejemplo_API_Sequential'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                            filepath= os.path.join(checkpoint_filepath + '.h5'),
                            monitor = 'val_loss',
                            mode='min',
                            verbose = 1,
                            save_best_only=True)

# %%
training_history = model.fit(x_train,
                             y_train,
                             validation_split=0.2,
                             epochs = 500,
                             batch_size = 100, 
                             callbacks = [tensorboard_callback, model_checkpoint_callback]) 

# %%
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.plot(training_history.history['mae'])
plt.plot(training_history.history['val_mae'])

plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'val'])

plt.subplot(1, 2, 2)

plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])

# %%
model_location = '../ModelsSaved/Ejemplo_API_Sequential.h5'

# %%
model_prueba = tf.keras.models.load_model(model_location)

# %%
model_prueba.input_shape

# %%
model_prueba.evaluate(x_test, y_test)

# %%
y_pred = model_prueba.predict(x_test)

r2_score(y_test, y_pred)

# %%
%load_ext tensorboard

# %%
%tensorboard --logdir seq_logs/ --port 6050
#http://localhost:6050/

# %%
model.evaluate(x_test, y_test)

# %%
y_pred = model.predict(x_test)

r2_score(y_test, y_pred)

# %%
"""
# Ejercicio
"""

# %%
"""
**Utilizando una Red Neuronal Profunda, resuelve el jercicio propuesto de una regresión lineal multiple del notebook 04_Ejercicios_00"**
"""

# %%
