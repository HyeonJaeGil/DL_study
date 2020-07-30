import tensorflow as tf
import cProfile
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot
import sys
import numpy as np
import os


# Data Augmentation

def corner_center_crop_reflect_v2(images, labels, crop_l):
    """
    Perform 4 corners and center cropping and reflection from images,
    resulting in 10x augmented patches.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, 10, h, w, C).
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:  # image.shape: (H, W, C)
        aug_image_orig = []
        # Crop image in 4 corners
        aug_image_orig.append(image[:crop_l, :crop_l])
        aug_image_orig.append(image[:crop_l, -crop_l:])
        aug_image_orig.append(image[-crop_l:, :crop_l])
        aug_image_orig.append(image[-crop_l:, -crop_l:])
        # Crop image in the center
        aug_image_orig.append(image[H // 2 - (crop_l // 2):H // 2 + (crop_l - crop_l // 2),
                              W // 2 - (crop_l // 2):W // 2 + (crop_l - crop_l // 2)])
        aug_image_orig = np.stack(aug_image_orig)  # (5, h, w, C)

        # Flip augmented images and add it
        aug_image_flipped = aug_image_orig[:, :, ::-1]  # (5, h, w, C)
        # aug_image = np.concatenate((aug_image_orig, aug_image_flipped), axis=0)    # (10, h, w, C)
        # augmented_images.append(aug_image)
        augmented_images.append(aug_image_flipped)

        if labels is not None:
            aug_labels = labels
            for i in range(10 - 1):
                aug_labels = np.concatenate((aug_labels, labels), axis=0)

    return np.stack(augmented_images), aug_labels  # shape: (N, 10, h, w, C)


def center_crop_v2(images, labels, crop_l):
    """
    Perform center cropping of images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3]
    cropped_images = []
    for image in images:  # image.shape: (H, W, C)
        # Crop image in the center
        cropped_images.append(image[H // 2 - (crop_l // 2):H // 2 + (crop_l - crop_l // 2),
                              W // 2 - (crop_l // 2):W // 2 + (crop_l - crop_l // 2)])
    return np.stack(cropped_images), labels


def random_crop_reflect_v2(images, labels, crop_l):
    """
    Perform random cropping and reflection from images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:  # image.shape: (H, W, C)
        # Randomly crop patch
        y = np.random.randint(H - crop_l)
        x = np.random.randint(W - crop_l)
        image = image[y:y + crop_l, x:x + crop_l]  # (h, w, C)

        # Randomly reflect patch horizontally
        reflect = bool(np.random.randint(2))
        if reflect:
            image = image[:, ::-1]

        augmented_images.append(image)
    return np.stack(augmented_images), labels  # shape: (N, h, w, C)


def augment_dataset_v2(x_train, y_train, Dataset_size, Batch_size, crop_l = 227):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # Shuffle and slice the dataset.
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(Batch_size, drop_remainder=False)
    augmented_images = np.empty(shape=(0, crop_l, crop_l, 3))
    augmented_labels = np.empty(shape=(0, 2))
    count = int(Dataset_size / Batch_size)
    for i in range(count):
        img_set, label_set = next(iter(train_dataset))
        img_set, label_set = img_set.numpy(), label_set.numpy()
        # size_list.append(img_set.shape)
        # lab_list.append(label_set.shape)
        center_crop_img_set = center_crop_v2(img_set, label_set, crop_l)
        crop_reflect_img_set = random_crop_reflect_v2(img_set, label_set, crop_l)

        # print(center_crop_img_set.shape, crop_reflect_img_set.shape)
        new_img_set = np.concatenate((center_crop_img_set[0], crop_reflect_img_set[0]), axis=0)
        new_label_set = np.concatenate((center_crop_img_set[1], crop_reflect_img_set[1]), axis=0)
        augmented_images = np.concatenate((augmented_images, new_img_set), axis=0)
        augmented_labels = np.concatenate((augmented_labels, new_label_set), axis=0)

    return augmented_images, augmented_labels
