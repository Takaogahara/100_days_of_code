__author__ = "Rafael Lopes Almeida"
__maintainer__ = "Rafael Lopes Almeida"
__email__ = "fael.rlopes@gmail.com"

__date__ = "23/07/2020"
__version__ = "0.0.1"

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# -----------------------------------------------------------------------------------

def get_mini_batch(image_generator):

    tuner_image, batch_label = next(image_generator)

    tuner_label = []
    for iter_tuner in range(len(batch_label)):
        if (batch_label[iter_tuner][0] == 0.0):
            tuner_label.append(1)
        else:
            tuner_label.append(0)
    tuner_label = np.array(tuner_label)

    return tuner_image, tuner_label

# tuner_image, tuner_label = utils.get_mini_batch(train_images_gen)

# -----------------------------------------------------------------------------------

def show_batch(image_batch, label_batch, CLASS_NAMES):
    plt.figure(figsize=(10,5))
    
    for img_num in range(5):
        plt.subplot(5,5,img_num+1)
        plt.imshow(image_batch[img_num])
        plt.title(CLASS_NAMES[label_batch[img_num]])
        plt.axis('off')
    
    plt.show()

# utils.show_batch(tuner_image, tuner_label, CLASS_NAMES)

# ----------------------------------------------------------------------------------- Model - Keras Tuner

def build_model(hp):
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters = hp.Int('conv_1_filter', min_value = 32, max_value = 64, step = 16),   
            kernel_size = hp.Choice('conv_1_kernel', values = [3,5,7]),                
            activation = 'relu',
            input_shape = (224, 224, 3)),
        keras.layers.MaxPooling2D(pool_size=(2)),

        keras.layers.Conv2D(
            filters = hp.Int('conv_2_filter', min_value = 64, max_value = 128, step = 16),
            kernel_size = hp.Choice('conv_2_kernel', values = [3,5,7]),
            activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(
            filters = hp.Int('conv_3_filter', min_value = 128, max_value = 256, step = 16),
            kernel_size = hp.Choice('conv_3_kernel', values = [3,5,7]),
            activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(
            filters = hp.Int('conv_4_filter', min_value = 256, max_value = 512, step = 16),
            kernel_size = hp.Choice('conv_4_kernel', values = [3,5,7]),
            activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(
            filters = hp.Int('conv_5_filter', min_value = 256, max_value = 512, step = 16),
            kernel_size = hp.Choice('conv_5_kernel', values = [1,2,3]),
            activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Dropout(0.4),

        keras.layers.Flatten(),
        keras.layers.Dense(
            units = hp.Int('dense_1_units', min_value = 32, max_value = 256, step = 16),
            activation = 'relu'),

        keras.layers.Dense(2, activation = 'sigmoid')])#softmax

    model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])),
        loss = 'binary_crossentropy', #sparse_categorical_crossentropy categorical_crossentropy binary_crossentropy 
        metrics = ['accuracy'])
    # keras.optimizers.SGD keras.optimizers.Adam

    return model

# ----------------------------------------------------------------------------------- Model

def build_model_test():
    model_test = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), input_shape = (224, 224, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size = (2,2)),

        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(256, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(512, (1,1), activation='relu'),
        # keras.layers.MaxPooling2D(pool_size = (2,2)),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(units=120, activation='relu'),
        keras.layers.Dense(units=2, activation='sigmoid')])

    model_test.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    # SGD Adam

    return model_test