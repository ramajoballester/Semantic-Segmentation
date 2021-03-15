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


model_name = 'resnet34_750'


gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


with open(os.path.join('pickle', 'val_x'), 'rb') as file:
    val_x = pickle.load(file)
    file.close()
    print('Val X loaded')
with open(os.path.join('pickle', 'val_y'), 'rb') as file:
    val_y = pickle.load(file)
    file.close()
    print('Val Y loaded')


model = tf.keras.models.load_model(os.path.join('trainings', 'models', model_name))

images_test = model.predict(val_x)
print(images_test.shape)
images_test = np.argmax(images_test, axis=3)
print(images_test.shape)
