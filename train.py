import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import visualkeras
import cv2
import os
import time
import numpy as np


def build_model(input_shape):
    inputs = Input(input_shape)

    conv_1 = Conv2D(16, (2, 2), padding='same')(inputs)
    conv_1 = LeakyReLU(0.2)(conv_1)
    conv_1 = Conv2D(16, (2, 2), padding='same')(conv_1)
    conv_1 = LeakyReLU(0.2)(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_1)

    conv_2 = Conv2D(32, (2, 2), padding='same')(pool_1)
    conv_2 = LeakyReLU(0.2)(conv_2)
    conv_2 = Conv2D(32, (2, 2), padding='same')(conv_2)
    conv_2 = LeakyReLU(0.2)(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_2)

    conv_3 = Conv2D(64, (2, 2), padding='same')(pool_2)
    conv_3 = LeakyReLU(0.2)(conv_3)
    conv_3 = Conv2D(64, (2, 2), padding='same')(conv_3)
    conv_3 = LeakyReLU(0.2)(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_3)

    conv_4 = Conv2D(128, (2, 2), padding='same')(pool_3)
    conv_4 = LeakyReLU(0.2)(conv_4)
    conv_4 = Conv2D(128, (2, 2), padding='same')(conv_4)
    conv_4 = LeakyReLU(0.2)(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_4)

    conv_5 = Conv2D(256, (2, 2), padding='same')(pool_4)
    conv_5 = LeakyReLU(0.2)(conv_5)
    conv_5 = Conv2D(256, (2, 2), padding='same')(conv_5)
    conv_5 = LeakyReLU(0.2)(conv_5)


    pool_5 = UpSampling2D(size=(2,2))(conv_5)
    concat_1 = tf.concat([pool_5, conv_4], axis=-1)
    conv_6 = Conv2DTranspose(384, (2, 2), padding='same')(concat_1)
    conv_6 = LeakyReLU(0.2)(conv_6)
    conv_6 = Conv2DTranspose(128, (2, 2), padding='same')(conv_6)
    conv_6 = LeakyReLU(0.2)(conv_6)

    pool_6 = UpSampling2D(size=(2,2))(conv_6)
    concat_2 = tf.concat([pool_6, conv_3], axis=-1)
    conv_7 = Conv2DTranspose(192, (2, 2), padding='same')(concat_2)
    conv_7 = LeakyReLU(0.2)(conv_7)
    conv_7 = Conv2DTranspose(64, (2, 2), padding='same')(conv_7)
    conv_7 = LeakyReLU(0.2)(conv_7)

    pool_7 = UpSampling2D(size=(2,2))(conv_7)
    concat_3 = tf.concat([pool_7, conv_2], axis=-1)
    conv_8 = Conv2DTranspose(96, (2, 2), padding='same')(concat_3)
    conv_8 = LeakyReLU(0.2)(conv_8)
    conv_8 = Conv2DTranspose(32, (2, 2), padding='same')(conv_8)
    conv_8 = LeakyReLU(0.2)(conv_8)

    pool_8 = UpSampling2D(size=(2,2))(conv_8)
    concat_4 = tf.concat([pool_8, conv_1], axis=-1)
    conv_9 = Conv2DTranspose(96, (2, 2), padding='same')(concat_4)
    conv_9 = LeakyReLU(0.2)(conv_9)
    conv_9 = Conv2DTranspose(32, (2, 2), padding='same')(conv_9)
    conv_9 = LeakyReLU(0.2)(conv_9)

    outputs = Conv2D(16, (2,2), padding='same', activation='sigmoid')(conv_9)

    model = tf.keras.Model(inputs, outputs)

    return model


# model = build_model((128,128,3))
# print(model.summary())
# visualkeras.layered_view(model, to_file='output.png')


def load_images(path):
    images_x = []
    images_y = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images_x.append(img[:, :int(img.shape[1]/2)])
            images_y.append(img[:, int(img.shape[1]/2):])
    return images_x, images_y


images_x, images_y = load_images(os.path.join('cityscapes_data', 'train'))
images_x = np.array(images_x)
images_y = np.array(images_y)
# cv2.imwrite('./image_x_1.jpg', cv2.cvtColor(images_x[0], cv2.COLOR_RGB2BGR))
# cv2.imwrite('./image_y_1.jpg', cv2.cvtColor(images_y[0], cv2.COLOR_RGB2BGR))

print(images_x.shape)

def unique_colors(images):
    colors = []
    for each_image in images:
        for i in range(each_image.shape[0]):
            for j in range(each_image.shape[1]):
                if not colors:
                    colors.append(each_image[i, j, :])
                else:
                    is_unique = True
                    for each_color in colors:
                        # print(each_image[i, j, :])
                        # print(each_color)
                        # time.sleep(2)
                        if each_image[i, j, 0] == each_color[0] and each_image[i, j, 1] == each_color[1] and each_image[i, j, 2] == each_color[2]:
                            is_unique = False
                    if is_unique:
                        colors.append(each_image[i, j])

    return colors

print('Loaded')
print(images_y[0,0,0,:])
# print(unique_colors(images_y[0:1]))

print(np.unique(images_y[:,:,:,1]))
print(np.unique(images_x[:,:,:,2]))
