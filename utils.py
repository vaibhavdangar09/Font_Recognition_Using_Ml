from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from PIL import ImageOps, Image
import os
import pathlib
from skimage import io
from tensorflow.keras import datasets, layers, models
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional,Dropout
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras


AUTOTUNE = tf.data.experimental.AUTOTUNE

# Loading and cropping the images in dataset to fixed dimensions of 100 * 100 from the given path
def crop_dataset(path):
    for fontlist in os.listdir(path):
        for font in os.listdir(os.path.join(path, fontlist)):
            path_new = os.path.join(path, fontlist, font)
            img = Image.open(path_new)
            img = ImageOps.fit(img, (100, 100), method=0, bleed=0.0, centering=(0.5, 0.5))
            img.save(path_new)

# Loading and counting the total number of images and classes in dataset
def list_dataset(path):
    data = pathlib.Path(path)
    image_count = len(list(data.glob('*/*.jpg')))
    CLASS_NAMES = np.array([item.name for item in data.glob('*') if item.name != "LICENSE.txt"])
    return data, image_count, CLASS_NAMES

# Preparing the dataset for training
def load_dataset(dir, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    data_gen = image_generator.flow_from_directory(directory=str(dir),
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                   classes=list(CLASS_NAMES))

    return data_gen

# Pre-processing the test image for prediction
def preprocess_test_image(path):
    img = io.imread(path, as_gray=False)
    crop_length = 100
    start_x = img.shape[0] / 2 - crop_length / 2
    start_y = img.shape[1] / 2 - crop_length / 2
    img = img[int(start_x): int(start_x + crop_length), int(start_y):int(start_y + crop_length)]
    img = np.divide(np.asarray(img), 255)
    return np.reshape(img, (1, 100, 100, 3))

# Defining input parameters required for the model
IMG_HEIGHT = 100
IMG_WIDTH = 100
OUTPUT_CLASSES = 100


from keras.layers import Reshape

# Initialize the model
model = models.Sequential()

# Add convolutional layers
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

# Add more convolutional layers
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(1024, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

# Reshape output of convolutional layers
model.add(Reshape((25, 1600)))  # Reshape to [25, 1600]

# Bidirectional LSTM layers
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)))
model.add(Bidirectional(LSTM(128, dropout=0.2)))

# Dense layers
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(OUTPUT_CLASSES, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()


# # Defining the CNN model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.4))
# model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.BatchNormalization())  # Add batch normalization here

# # Remove Flatten layer
# model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)))  # Add Bidirectional LSTM layer
# model.add(Bidirectional(LSTM(128, dropout=0.2)))  # Add Bidirectional LSTM layer

# model.add(layers.Dense(500, activation='relu'))
# model.add(layers.Dropout(0.4))
# model.add(layers.Dense(OUTPUT_CLASSES, activation='softmax'))








# inputs = Input(shape=(100,100,1))
# s = Lambda(lambda x: x / 255, output_shape=lambda x: x)(inputs)


# # convolution layer with kernel size (3,3)
# conv_1 = Conv2D(16, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(s)
# conv_1 = Dropout(0.25)(conv_1)
# conv_1 = Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(conv_1)
# # poolig layer with kernel size (2,2)
# pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
# conv_2 = Conv2D(32, (3,3), activation = 'relu',kernel_initializer='he_normal' , padding='same')(pool_1)
# conv_2= BatchNormalization(axis=-1)(conv_2)
# conv_2 = Dropout(0.25)(conv_2)
# conv_2 = Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(conv_2)
# pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
# conv_3 = Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(pool_2)
# conv_3= BatchNormalization(axis=-1)(conv_3)
# conv_3 = Dropout(0.25)(conv_3)
# conv_3 = Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(conv_3)
# conv_4 = Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(conv_3)
# # poolig layer with kernel size (2,1)
# pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
# conv_5 = Conv2D(256, (3,3), activation = 'relu',kernel_initializer='he_normal' , padding='same')(pool_4)
# # Batch normalization layer
# batch_norm_5 = BatchNormalization()(conv_5)
 
# conv_6 = Conv2D(256, (3,3), activation = 'relu',kernel_initializer='he_normal' , padding='same')(batch_norm_5)
# batch_norm_6 = BatchNormalization()(conv_6)
# pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

# conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
# squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
# squeezed = Lambda(lambda x: K.squeeze(x, axis=1), output_shape=lambda x: (x[0], x[2], x[3]))(conv_7)
# # bidirectional LSTM layers with units=128
# blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
# blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
 
# outputs = Dense(OUTPUT_CLASSES, activation = 'softmax')(blstm_2)

# # model to be used at test time
# act_model = Model(inputs, outputs)

model.summary()