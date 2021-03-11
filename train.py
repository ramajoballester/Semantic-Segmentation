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
from utils import *

train_x = load_images(os.path.join('cityscapesScripts', 'leftImg8bit', 'train', 'aachen'), (256,256))
train_y = load_images(os.path.join('cityscapesScripts', 'gtFine', 'train_labels', 'aachen'), (256, 256))
val_x = load_images(os.path.join('cityscapesScripts', 'leftImg8bit', 'val', 'aachen'), (256,256))
val_y = load_images(os.path.join('cityscapesScripts', 'gtFine', 'train_labels', 'aachen'), (256, 256))

train_x = np.array(train_x)
train_y = np.array(train_y)[:,:,:,0]
val_x = np.array(val_x)
val_y = np.array(val_y)[:,:,:,0]

model = build_model(train_x[0].shape, len(labels))

loss = tf.keras.losses.CategoricalCrossentropy()
metrics = tf.keras.metrics.categorical_accuracy
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])




tb_callback = TensorboardCallback('training')
ckpt_callback = tf.keras.callbacks.ModelCheckpoint('models', monitor='val_categorical_accuracy',
                                                    verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto',
                                                    save_freq='epoch')

model.fit(train_x, tf.one_hot(train_y, depth=len(labels)),
            validation_data=(val_x, val_y), epochs=10000,
            callbacks=[tb_callback, ckpt_callback], batch_size=16)
