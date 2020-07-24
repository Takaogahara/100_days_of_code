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
test_path = ('D:/Code/Source/Semiconductor CV/Images/Test/')

# ----------------------------------------------------------------------------------- Parameters

image_count_train = sum(len(files) for _, _, files in os.walk(train_path))
image_count_test = sum(len(files) for _, _, files in os.walk(test_path))
CLASS_NAMES = np.array([name for name in os.listdir(train_path) 
                                if os.path.isdir(general_path)])

BATCH_SIZE = 16
TARGET_SIZE = (224, 224)
STEPS_PER_EPOCH_TRAIN = np.ceil(image_count_train/BATCH_SIZE)
STEPS_PER_EPOCH_TEST = np.ceil(image_count_test/BATCH_SIZE)

# ----------------------------------------------------------------------------------- IMG Generator

datagen = ImageDataGenerator(rescale=1/255)

train_images_gen = datagen.flow_from_directory(
        train_path, 
        batch_size=BATCH_SIZE, target_size=TARGET_SIZE, 
        shuffle=True, classes=list(CLASS_NAMES))


test_images_gen = datagen.flow_from_directory(
        test_path, batch_size=BATCH_SIZE, 
        target_size=TARGET_SIZE, shuffle=True)

# -----------------------------------------------------------------------------------

tuner_image, tuner_label = utils.get_mini_batch(train_images_gen)
# utils.show_batch(tuner_image, tuner_label, CLASS_NAMES)

# ----------------------------------------------------------------------------------- Keras Tuner

tuner_search = RandomSearch(
        utils.build_model, max_trials=3, 
        objective='val_accuracy',project_name='Kerastuner', 
        directory='D:/Code/Source/Semiconductor CV/')


tuner_search.search(
        train_images_gen, steps_per_epoch=250,
        epochs=3, verbose=1,
        validation_data = test_images_gen,)

model = tuner_search.get_best_models(num_models=1)[0]
model.summary()

# -----------------------------------------------------------------------------------

model.save_weights('D:/Code/Source/Semiconductor CV/CNN_weights.h5')

# -----------------------------------------------------------------------------------

# model.fit(train_images_gen,epochs=5,
#         steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
#         validation_data=test_images_gen,
#         validation_steps=STEPS_PER_EPOCH_TEST)
 