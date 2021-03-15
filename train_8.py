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
from unet import build_model
import pickle
import segmentation_models as sm

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)



gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():


train_limit = None
val_limit = None

with open(os.path.join('pickle', 'train_x'), 'rb') as file:
    train_x = pickle.load(file)
    file.close()
    print('Train X loaded')
with open(os.path.join('pickle', 'train_y_8'), 'rb') as file:
    train_y = pickle.load(file)
    file.close()
    print('Train Y loaded')
with open(os.path.join('pickle', 'val_x'), 'rb') as file:
    val_x = pickle.load(file)
    file.close()
    print('Val X loaded')
with open(os.path.join('pickle', 'val_y_8'), 'rb') as file:
    val_y = pickle.load(file)
    file.close()
    print('Val Y loaded')


# train_x = load_images(os.path.join('cityscapesScripts', 'leftImg8bit', 'train'), (256,256))
# train_y = load_images(os.path.join('cityscapesScripts', 'gtFine', 'train_labels'), (256, 256))
# val_x = load_images(os.path.join('cityscapesScripts', 'leftImg8bit', 'val'), (256,256))
# val_y = load_images(os.path.join('cityscapesScripts', 'gtFine', 'val_labels'), (256, 256))




# train_x = np.array(train_x)
# train_y = np.array(train_y)[:,:,:,0]
# val_x = np.array(val_x)
# val_y = np.array(val_y)[:,:,:,0]

# model = build_model(train_x[0].shape, len(labels))
# model = build_model(nx=256, ny=256, channels=3, num_classes=35, layer_depth=5,
#                     filters_root=64, kernel_size=3, pool_size=2, dropout_rate=0.2)

model = sm.Unet('resnet34', input_shape=train_x[0].shape, classes=8)

loss = tf.keras.losses.CategoricalCrossentropy()
metrics = tf.keras.metrics.categorical_accuracy
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

tb_logs_dir, save_models_dir, data_train_dir = create_datetime_dirs(os.getcwd())
tb_callback = TensorboardCallback(tb_logs_dir)
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(save_models_dir, monitor='val_categorical_accuracy',
                                                    verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto',
                                                    save_freq='epoch')

if train_limit:
    train_x = train_x[:train_limit]
    train_y = train_y[:train_limit]
if val_limit:
    val_x = val_x[:val_limit]
    val_y = val_y[:val_limit]


model.fit(train_x, tf.one_hot(train_y, depth=8),
            validation_data=(val_x, tf.one_hot(val_y, depth = 8)), epochs=10000,
            callbacks=[tb_callback, ckpt_callback], batch_size=1)
