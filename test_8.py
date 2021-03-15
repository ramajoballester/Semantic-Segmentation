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

model_name = 'resnet34_8'

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():

model = tf.keras.models.load_model(os.path.join('trainings', 'models', model_name))
print('Model loaded')

# almeria_images = np.array(load_images(os.path.join('videos', 'video2'), (256, 256)))
# print('Images loaded')

with open(os.path.join('pickle', 'val_x'), 'rb') as file:
    val_x = pickle.load(file)
    file.close()
    print('Val X loaded')

almeria_preds = model.predict(val_x, batch_size=8)
almeria_preds = np.array(np.argmax(almeria_preds, axis=-1))
print(almeria_preds.shape)
print('Images predicted')

catids = np.array(range(8))
colors = np.array([labels[0].color, labels[7].color, labels[11].color,
                    labels[17].color, labels[21].color, labels[23].color,
                    labels[24].color, labels[26].color])

almeria_color = catid2image(almeria_preds, catids, colors)
print(almeria_color.shape)
print('Images converted')

i = 0
path = os.path.join('videos', 'val_x_pred')
for each_image in almeria_color:
    i += 1
    r, g, b = cv2.split(each_image)
    each_image[:,:,0] = b
    each_image[:,:,1] = g
    each_image[:,:,2] = r
    img = np.array(each_image, dtype='uint8')
    img = cv2.resize(img, (1920, 1080))
    cv2.imwrite(os.path.join(path, '{:0>5}'.format(i) + '.jpg'), img)

path = os.path.join('videos', 'val_x')
for each_image in val_x:
    i += 1
    r, g, b = cv2.split(each_image)
    each_image[:,:,0] = b
    each_image[:,:,1] = g
    each_image[:,:,2] = r
    img = np.array(each_image, dtype='uint8')
    img = cv2.resize(img, (1920, 1080))
    cv2.imwrite(os.path.join(path, '{:0>5}'.format(i) + '.jpg'), img)
