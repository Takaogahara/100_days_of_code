__author__ = "Rafael Lopes Almeida"
__maintainer__ = "Rafael Lopes Almeida"
__email__ = "fael.rlopes@gmail.com"

__date__ = "23/07/2020"
__version__ = "0.0.1"

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

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

        keras.layers.Dense(1, activation = 'sigmoid')])#softmax

    model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])),
        loss = 'binary_crossentropy', #sparse_categorical_crossentropy categorical_crossentropy binary_crossentropy 
        metrics = ['accuracy'])
    # keras.optimizers.SGD keras.optimizers.Adam

    return model

# ----------------------------------------------------------------------------------- Model

def build_model_fix():
    model_fix = keras.Sequential([
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
        keras.layers.Dense(units=1, activation='sigmoid')])

    model_fix.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    # SGD Adam

    return model_fix

# ----------------------------------------------------------------------------------- 

def generate_model(model_type = 'single', 
    folder_path = 'D:/Code/Source/Semiconductor CV/h5/', 
    file_name = 'fit_weights.h5'):

    if (model_type == 'load'):
        model = keras.models.load_model('D:/Code/Source/Semiconductor CV/h5/'+str(file_name))

    elif (model_type == 'single'):
        model = build_model_fix()
        model.save(str(folder_path)+str(file_name))
    
    else:
        print('model not found')

    return model

def plot_history(results_fit, save_path = 'D:/Code/Source/Semiconductor CV/'):
    plt.plot(results_fit.history['accuracy'])
    plt.plot(results_fit.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(save_path)+'train_test_acc.png')

    plt.show()