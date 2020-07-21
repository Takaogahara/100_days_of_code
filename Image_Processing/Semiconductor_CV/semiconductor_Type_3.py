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

print(f'Tensorflow version: {tf.__version__}')
print(f'Keras version: {keras.__version__}')

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

print(f'TF config completed')

# -----------------------------------------------------------------------------------

general_path = ('D:/Code/Source/Semiconductor CV/Images/')
train_path = ('D:/Code/Source/Semiconductor CV/Images/Train/')
test_path = ('D:/Code/Source/Semiconductor CV/Images/Test/')

# -----------------------------------------------------------------------------------

image_count = sum(len(files) for _, _, files in os.walk(train_path))
CLASS_NAMES = np.array([name for name in os.listdir(train_path) 
                                if os.path.isdir(general_path)])

BATCH_SIZE = 5
TARGET_SIZE = (224, 224)
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

# -----------------------------------------------------------------------------------

datagen = ImageDataGenerator(rescale=1./255)

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

# -----------------------------------------------------------------------------------

tuner_search = RandomSearch(
                    utils.build_model, max_trials=5, 
                    objective='val_accuracy',project_name='Kerastuner', 
                    directory='D:/Code/Source/Semiconductor CV/')

with tf.device('/GPU:0'):
    tuner_search.search(
                        tuner_image, tuner_label, 
                        epochs=3,validation_split=0.1, 
                        verbose=2)


# model = tuner_search.get_best_models(num_models=1)[0]
# model.summary()

# -----------------------------------------------------------------------------------

# model.save_weights('D:/Code/Source/Semiconductor CV/CNN_weights.h5')
