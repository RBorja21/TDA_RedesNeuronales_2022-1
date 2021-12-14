# %%
"""
# Introducción a la Redes Neuronales Convolucionales (CNN´s)
"""

# %%
"""
## Clasificación de Imágenes 
"""

# %%
#!pip3 install opencv-python 

# %%
from random import randint 

import cv2
import os 
import time

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

# %%
import tensorflow as tf 

from tensorflow import keras 
from tensorflow.keras import layers 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
"""
## CIFAR-10
"""

# %%
"""
El conjunto de datos CIFAR-10 consta de 60000 imágenes en color de 32x32 en 10 clases, con 6000 imágenes por clase. Hay 50000 imágenes de entrenamiento y 10000 imágenes de prueba.
"""

# %%
"""
`https:www.cs.toronto.edu/~kriz/cifar.html`
"""

# %%
cifer_10 = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifer_10.load_data()

# %%
train_images.shape, test_images.shape

# %%
train_images[0].shape

# %%
train_images[0]

# %%
train_labels[:20]

# %%
lookup = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

# %%
def show_img(images, labels, n_images):
    
    random_int = randint(0, labels.shape[0] - n_images)
    
    imgs, labels = images[random_int : random_int + n_images], \
                   labels[random_int : random_int + n_images]
        
    _, figs = plt.subplots(1, n_images, figsize=(n_images * 3, 3))
        
    for fig, img, label in zip(figs, imgs, labels):
        fig.imshow(img)
        ax = fig.axes
        
        ax.set_title(lookup[int(label)])
        
        ax.title.set_fontsize(20)
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

# %%
show_img(train_images, train_labels, 3)

# %%
show_img(train_images, train_labels, 5)

# %%
#os.mkdir('../DataSets/cifar')
#os.mkdir('../DataSets/cifar/train')
#os.mkdir('../DataSets/cifar/test')

# %%
train_dir = '../DataSets/cifar/train/'
test_dir = '../DataSets/cifar/test/'

# %%
i = 0

for img, label in zip(train_images, train_labels):
    
    path = train_dir + str(lookup[int(label)])
    
    cv2.imwrite(os.path.join(path, str(i) + '.jpeg'), img)
    i += 1
    #cv2.waitKey(0)
    

# %%
i = 0

for img, label in zip(test_images, test_labels):
    
    path = test_dir + str(lookup[int(label)])
    
    cv2.imwrite(os.path.join(path, str(i) + '.jpeg'), img)
    
    i += 1
    #cv2.waitKey(0)

# %%
train_image_generator = ImageDataGenerator(rescale=1./255)

test_image_generator = ImageDataGenerator(rescale=1./255)

# %%
batch_size = 64

# %%
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                          directory=train_dir,
                                                          shuffle=True,
                                                          target_size=(32, 32))

# %%
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                          directory=test_dir,
                                                          shuffle=True,
                                                          target_size=(32, 32))

# %%
sample_batch = next(train_data_gen)
sample_batch[0].shape

# %%
"""
 **Las imágenes del generador de datos utilizadas para entrenar un modelo de TF se especifican como un tensor de cuatro dimensiones**
 
 - **La primera dimensión representa el número de imágenes en un lote**
 - **Las siguientes dos dimensiones representan la `altura` y el `ancho` de cada imagen**
 - **Lo último se refiere a la cantidad canales para una imagen individual**
"""

# %%
conv_model = tf.keras.models.Sequential([
           
    layers.Conv2D(16, (3, 3), padding='same', activation='relu',
                 input_shape=sample_batch[0].shape[1:]),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# %%
conv_model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

# %%
conv_model.summary()

# %%
start = time.time()

training_hist = conv_model.fit(train_data_gen,
                              epochs=10,
                               #número de pasos por época es igual a la longitud de las imágenes
                               # de entrenamiento dividido por el tamaño del lote.
                              steps_per_epoch=len(train_images) // batch_size,
                              validation_data=test_data_gen,
                              validation_steps=len(test_images) // batch_size)

end = time.time()
print("Time: ", end - start, "seg")
t_total = (end - start)*(1/60)
print("Time: ", t_total, "min")

# %%
acc = training_hist.history['accuracy']
val_acc = training_hist.history['val_accuracy']

loss = training_hist.history['loss']
val_loss = training_hist.history['val_loss']

epoch_range = range(10)

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)

plt.plot(epoch_range, acc, label='Training accuracy')
plt.plot(epoch_range, val_acc, label ='Validation Accuracy')

plt.legend(loc='lower right')
plt.title('Training and validation Accuracy')

plt.subplot(1, 2, 2)

plt.plot(epoch_range, loss, label='Training Loss')
plt.plot(epoch_range, val_loss, label='Validation Loss')

plt.legend(loc='lower left')
plt.title('Training and Validation Loss')

plt.show()

# %%
"""
**Training Accuracy es mejor que Validation Accuracy, lo que indica que el modelo podría estar demasiado ajustado en los datos del entrenamiento.**
"""

# %%
from tensorflow.keras.preprocessing import image 

# %%
test_images[0].shape

# %%
def perform_test(model, img, label):
    plt.imshow(img)
    
    test_img = np.expand_dims(img, axis=0)
    result = model.predict(test_img)
    
    print('Actual label: ', lookup[int(label)])
    print('Predicted label: ', lookup[np.argmax(result)])

# %%
perform_test(conv_model, test_images[0], test_labels[0])

# %%
perform_test(conv_model, test_images[10], test_labels[10])

# %%
perform_test(conv_model, test_images[53], test_labels[53])

# %%
perform_test(conv_model, test_images[99], test_labels[99])

# %%
perform_test(conv_model, test_images[147], test_labels[147])

# %%
perform_test(conv_model, test_images[631], test_labels[631])

# %%
"""
# Using Image Transformation and Dropout
"""

# %%
image_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=.05,
    height_shift_range=.05,
    horizontal_flip=True,
    zoom_range=0.3
)

train_data_gen_aug = image_gen.flow_from_directory(batch_size=batch_size,
                                                          directory=train_dir,
                                                          shuffle=True,
                                                          target_size=(32, 32))

# %%
plt.imshow(train_data_gen_aug[21][0][0])

# %%
conv_model_with_dropout = tf.keras.models.Sequential([
           
    layers.Conv2D(16, (3, 3), padding='same', activation='relu',
                 input_shape=sample_batch[0].shape[1:]),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    
    layers.Dropout(0.30),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    
    layers.Dropout(0.40),
    
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# %%
conv_model_with_dropout.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

# %%
start = time.time()

training_hist_wd = conv_model_with_dropout.fit(train_data_gen,
                              epochs=10,
                              steps_per_epoch=len(train_images) // batch_size,
                              validation_data=test_data_gen,
                              validation_steps=len(test_images) // batch_size)


end = time.time()
print("Time: ", end - start, "seg")
t_total = (end - start)*(1/60)
print("Time: ", t_total, "min")

# %%
acc_wd = training_hist_wd.history['accuracy']
val_acc_wd = training_hist_wd.history['val_accuracy']

loss_wd = training_hist_wd.history['loss']
val_loss_wd = training_hist_wd.history['val_loss']

epoch_range_wd = range(10)

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)

plt.plot(epoch_range_wd, acc_wd, label='Training accuracy')
plt.plot(epoch_range_wd, val_acc_wd, label ='Validation Accuracy')

plt.legend(loc='lower right')
plt.title('Training and validation Accuracy')

plt.subplot(1, 2, 2)

plt.plot(epoch_range_wd, loss_wd, label='Training Loss')
plt.plot(epoch_range_wd, val_loss_wd, label='Validation Loss')

plt.legend(loc='lower left')
plt.title('Training and Validation Loss')

plt.show()

# %%
perform_test(conv_model_with_dropout, test_images[1], test_labels[1])

# %%
perform_test(conv_model_with_dropout, test_images[100], test_labels[100])

# %%
perform_test(conv_model_with_dropout, test_images[500], test_labels[500])

# %%
perform_test(conv_model_with_dropout, test_images[890], test_labels[890])

# %%
perform_test(conv_model_with_dropout, test_images[591], test_labels[591])

# %%
