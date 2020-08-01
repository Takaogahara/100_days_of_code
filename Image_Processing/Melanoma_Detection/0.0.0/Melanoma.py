import numpy as np
import os
import pydicom
import utils

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# ----------------------------------------------------------------------------------- Folder Path

general_path = ('D:/Code/Source/Melanoma/imagens/')

train_path = ('D:/Code/Source/Melanoma/imagens/train/')
# val_path = ('D:/Code/Source/Melanoma/imagens/train/')
test_path = ('D:/Code/Source/Melanoma/imagens/test/')

# filename = 'D:/Code/Source/Melanoma/train/'+'ISIC_0015719.dcm'
# ds = pydicom.dcmread(filename)

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


train_images_gen = datagen_train.flow_from_directory(
        train_path, 
        batch_size=BATCH_SIZE, class_mode='binary',
        target_size=TARGET_SIZE, shuffle=True)