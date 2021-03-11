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


def build_model(input_shape, output_classes):
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

    outputs = Conv2D(output_classes, (2,2), padding='same', activation='softmax')(conv_9)

    model = tf.keras.Model(inputs, outputs)

    return model


def get_dirs_files(path):
    items = os.listdir(path)
    dirs = []
    files = []
    for item in items:
        if os.path.isfile(os.path.join(path, item)):
            files.append(item)
        elif os.path.isdir(os.path.join(path, item)):
            dirs.append(item)

    return dirs, files


# Loads all images recursively under path for one sublevel
def load_images(path, img_resolution=None):
    dirs, files = get_dirs_files(path)
    if not dirs:
        dirs = ['.']

    images = []
    dirs.sort()
    for each_dir in dirs:
        img_names = os.listdir(os.path.join(path, each_dir))
        img_names.sort()
        for each_img_name in img_names:
            img = cv2.imread(os.path.join(path, each_dir, each_img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img_resolution:
                img = cv2.resize(img, img_resolution, interpolation=cv2.INTER_NEAREST)
            images.append(img)

    return images


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


def image2id(images, labels):
    images_ids = []
    for each_image in images:
        for i in range(each_image.shape[0]):
            for j in range(each_image.shape[1]):
                color = each_image[i,j]
                idx = 0
                for k in range(len(labels)):
                    if np.array_equal(color, labels[k].color):
                        idx = k
                each_image[i,j,0] = labels[idx].categoryId
        images_ids.append(each_image[:,:,0])
    return images_ids


def id2image(images_ids, ids, colors):
    images = copy.deepcopy(images_ids)
    images = np.expand_dims(images, axis=-1)
    images = np.repeat(images, 3, axis=-1)
    ids = np.array(ids)

    for i in range(len(colors)):
        images[images_ids==ids[i]] = np.array(colors[i])

    return images


class TensorboardCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()

    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar('Train/Loss', logs['loss'], epoch)
        tf.summary.scalar('Train/Accuracy', logs['categorical_accuracy'], epoch)
        tf.summary.scalar('Val/Loss', logs['val_loss'], epoch)
        tf.summary.scalar('Val/Accuracy', logs['val_categorical_accuracy'], epoch)
