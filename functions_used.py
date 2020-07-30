import tensorflow as tf
import cProfile
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot
import sys
import numpy as np
import os


def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    (train_images , test_images) = (train_images / 255.0, test_images / 255.0)

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


def load_cifar_10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    (train_images, test_images) = (train_images / 255.0, test_images / 255.0)

    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
    test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)


    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


def summarize_diagnostics(history):
    fig, loss_ax = pyplot.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(history.history['acc'], 'b', label='train acc')
    acc_ax.plot(history.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    # pyplot.show()
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def define_model_mnist(learning_rate):
    model = tf.keras.Sequential()
    # L1
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # L2
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # L3 fully connected
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                     metrics=['accuracy'])
    model.summary()
    return model


def define_model_cifar10(learning_rate):

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                               padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer='l2'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    model.summary()
    return model


def define_alexnet_keras(learning_rate):

    model = tf.keras.Sequential()
    #L1
    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), input_shape=(227,227,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))

    #L2
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.5))

    #L3
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))

    #L4
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(tf.keras.layers.Dropout(0.5))

    #L5
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.5))

    #L6 Fully Connected
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    #L7 Fully Connected
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    #L8 Fully Connected
    model.add(tf.keras.layers.Dense(2, kernel_initializer='glorot_normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    model.summary()

    return model

def define_alexnet_keras_rev(learning_rate):

    model = tf.keras.Sequential()
    #L1
    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), input_shape=(227,227,3)))
    #L2
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    #L3
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #L4
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    #L6 Fully Connected
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    #L7 Fully Connected
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    #L8 Fully Connected
    model.add(tf.keras.layers.Dense(2, kernel_initializer='glorot_normal', activation='softmax'))
    optimizer = optimizers.SGD(lr=learning_rate, decay=5e-5, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model


# Data Augmentation

def corner_center_crop_reflect(images, crop_l, labels = None):
    """
    Perform 4 corners and center cropping and reflection from images,
    resulting in 10x augmented patches.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, 10, h, w, C).
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        aug_image_orig = []
        # Crop image in 4 corners
        aug_image_orig.append(image[:crop_l, :crop_l])
        aug_image_orig.append(image[:crop_l, -crop_l:])
        aug_image_orig.append(image[-crop_l:, :crop_l])
        aug_image_orig.append(image[-crop_l:, -crop_l:])
        # Crop image in the center
        aug_image_orig.append(image[H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                                    W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
        aug_image_orig = np.stack(aug_image_orig)    # (5, h, w, C)

        # Flip augmented images and add it
        aug_image_flipped = aug_image_orig[:, :, ::-1]    # (5, h, w, C)
        aug_image = np.concatenate((aug_image_orig, aug_image_flipped), axis=0)    # (10, h, w, C)
        augmented_images.append(aug_image)

        if labels is not None:
            aug_labels = labels
            for i in range(10-1):
                aug_labels = np.concatenate((aug_labels, labels), axis=0)

    return np.stack(augmented_images), aug_labels    # shape: (N, 10, h, w, C)


def center_crop(images, crop_l):
    """
    Perform center cropping of images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3]
    cropped_images = []
    for image in images:    # image.shape: (H, W, C)
        # Crop image in the center
        cropped_images.append(image[H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                              W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
    return np.stack(cropped_images)


def random_crop_reflect(images, crop_l):
    """
    Perform random cropping and reflection from images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        # Randomly crop patch
        y = np.random.randint(H-crop_l)
        x = np.random.randint(W-crop_l)
        image = image[y:y+crop_l, x:x+crop_l]    # (h, w, C)

        # Randomly reflect patch horizontally
        reflect = bool(np.random.randint(2))
        if reflect:
            image = image[:, ::-1]

        augmented_images.append(image)
    return np.stack(augmented_images)    # shape: (N, h, w, C)


def augment_dataset(raw_images, raw_labels, crop_l=227):

    #Center cropped set
    center_cropped_images = center_crop(raw_images, crop_l)
    center_cropped_labels = raw_labels

    #crop_reflected_set
    # crop_reflected_images = random_crop_reflect(raw_images, crop_l)
    # crop_reflected_labels = raw_labels

    #4 corners, center cropped_and_reflected set
    # corner_center_cropped_image, corner_center_cropped_label = corner_center_crop_reflect(raw_images, crop_l, raw_labels)
    # corner_center_cropped_image = corner_center_cropped_image.reshape(-1, crop_l, crop_l, 3)

    # print(center_cropped_images.shape, crop_reflected_images.shape, corner_center_cropped_image.shape,
    #       corner_center_cropped_label.shape)

    # aug_images = np.concatenate([center_cropped_images,
    #                               crop_reflected_images, corner_center_cropped_image], axis=0)
    # aug_images = np.concatenate([center_cropped_images, crop_reflected_images], axis=0)
    # aug_labels = np.concatenate([center_cropped_labels, crop_reflected_labels, corner_center_cropped_label], axis=0)
    # aug_labels = np.concatenate([center_cropped_labels, crop_reflected_labels], axis=0)
    # return aug_images, aug_labels
    # return corner_center_cropped_image, corner_center_cropped_label
    return center_cropped_images, center_cropped_labels

def train_and_evaluate_model(model, train_images, train_labels,
                             test_images, test_labels, batch_size, training_epoch):
    history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=training_epoch,
                            validation_data=(test_images, test_labels), shuffle=True)
    evaluation = model.evaluate(test_images, test_labels)
    print('loss: ', evaluation[0])
    print('accuracy', evaluation[1])

    return history


if __name__ == '__main__':
    pass
