import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import visualkeras
import cv2
import os
import time
import numpy as np
import copy
from cityscapesscripts.helpers.labels import labels


train_x = load_images(os.path.join('cityscapesScripts', 'leftImg8bit', 'train', 'aachen'), (256,256))
train_y = load_images(os.path.join('cityscapesScripts', 'gtFine', 'train_labels', 'aachen'), (256, 256))
train_x = np.array(train_x)
train_y = np.array(train_y[:,:,:,0])

model = build_model(images_y[0].shape, len(labels))

loss = tf.keras.losses.CategoricalCrossentropy()
metrics = tf.keras.metrics.categorical_accuracy
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

model.fit(train_x, tf.one_hot(train_y, depth=len(labels)),
            validation_split=0.7, epochs=100)
