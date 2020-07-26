__author__ = "Rafael Lopes Almeida"
__maintainer__ = "Rafael Lopes Almeida"
__email__ = "fael.rlopes@gmail.com"

__date__ = "23/07/2020"
__version__ = "0.0.1"

import utils
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# from PIL import Image
# import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# ----------------------------------------------------------------------------------- Folder Path

general_path = ('D:/Code/Source/Semiconductor CV/Images/')

train_path = ('D:/Code/Source/Semiconductor CV/Images/Train/')
val_path = ('D:/Code/Source/Semiconductor CV/Images/Validation/')
test_path = ('D:/Code/Source/Semiconductor CV/Images/Test/')

# ----------------------------------------------------------------------------------- Parameters

image_count_train = sum(len(files) for _, _, files in os.walk(train_path))
image_count_test = sum(len(files) for _, _, files in os.walk(test_path))

BATCH_SIZE = 16
TARGET_SIZE = (224, 224)
STEPS_PER_EPOCH = np.ceil(image_count_train/BATCH_SIZE)

# ----------------------------------------------------------------------------------- IMG Generator

datagen_train = ImageDataGenerator(rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

datagen = ImageDataGenerator(rescale=1/255)

# ----------------------------------------------------------------------------------- Batch Flow from Directory

train_images_gen = datagen_train.flow_from_directory(
        train_path, 
        batch_size=BATCH_SIZE, target_size=TARGET_SIZE, 
        shuffle=True) #class_mode='binary'

val_images_gen = datagen.flow_from_directory(
        val_path,
        batch_size=BATCH_SIZE, #class_mode='binary'
        target_size=TARGET_SIZE, shuffle=True)

test_images_gen = datagen.flow_from_directory(
        test_path, 
        batch_size=BATCH_SIZE, #class_mode='binary'
        target_size=TARGET_SIZE, shuffle=True)

# ----------------------------------------------------------------------------------- Build Model
# ------------------------------------ Load Model

# model = keras.models.load_model('D:/Code/Source/Semiconductor CV/h5/fit_weights.h5')
# model.summary()

# ------------------------------------ Normal Model

model = utils.build_model_test()
model.save('D:/Code/Source/Semiconductor CV/h5/Single_weights.h5')
model.summary()

# ------------------------------------ Keras Tuner

# tuner_search = RandomSearch(
#         utils.build_model, max_trials = 5, 
#         objective ='val_accuracy',project_name = 'Kerastuner', 
#         directory ='D:/Code/Source/Semiconductor CV/')

# tuner_search.search(
#         train_images_gen, steps_per_epoch = 150,
#         epochs = 3, verbose = 1,
#         validation_data = val_images_gen)

# model = tuner_search.get_best_models(num_models=1)[0]
# model.save('D:/Code/Source/Semiconductor CV/h5/KerasTuner_weights.h5')
# model.summary()

# ----------------------------------------------------------------------------------- Fit Model

model.fit(train_images_gen, epochs = 5,
        steps_per_epoch = 150,
        validation_data = val_images_gen)

model.save('D:/Code/Source/Semiconductor CV/h5/fit_weights.h5')
model.summary()

# ----------------------------------------------------------------------------------- Predict Model

prediction_prob = model.predict(test_images_gen)

print(prediction_prob)

x = 25