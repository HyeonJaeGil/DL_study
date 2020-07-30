from functions_used import load_cifar_10, define_model_cifar10, train_and_evaluate_model, summarize_diagnostics,\
    define_alexnet_keras, random_crop_reflect, center_crop, corner_center_crop_reflect, augment_dataset
from data_preprocess import augment_dataset_v2
from dataset import asirra as dataset
import tensorflow as tf
import numpy as np
from keras_preprocessing.image import load_img, img_to_array, ImageDataGenerator
import os

if __name__ == '__main__':

    # Parameter Setting
    learning_rate = 0.01
    training_epoch = 10
    batch_size = 32
    display_step = 20
    data_aug = True

    root_dir = os.path.join("/home/hj", "Downloads", "asirra")
    trainval_dir = os.path.join(root_dir, "train")
    print(trainval_dir)

    print(tf.__version__)
    # with Keras

    # load the dataset
    trainval_images, trainval_labels = dataset.read_asirra_subset(trainval_dir, one_hot=True)

    trainval_size = trainval_images.shape[0]
    val_size = int(trainval_size * 0.2)

    raw_train_images, raw_train_labels = trainval_images[val_size:], trainval_labels[val_size:]
    raw_val_images, raw_val_labels = trainval_images[:val_size], trainval_labels[:val_size]

    # train_dataset = tf.data.Dataset.from_tensor_slices((raw_train_images, raw_train_labels))
    # # Shuffle and slice the dataset.
    # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(2)
    #
    # print(list(train_dataset.as_numpy_iterator()))

    # datagenerator = ImageDataGenerator(horizontal_flip=True, rotation_range=90, zoom_range=[0.5, 1.0])
    # iterator = datagenerator.flow(raw_train_images, raw_train_labels)

    # # augument the dataset(12 times bigger)
    # aug_train_images, aug_train_labels = augment_dataset(raw_train_images, raw_train_labels, crop_l=227)
    # aug_val_images, aug_val_labels = augment_dataset(raw_val_images, raw_val_labels, crop_l=227)
    aug_train_images, aug_train_labels = augment_dataset_v2(raw_train_images, raw_train_labels, Dataset_size=(trainval_size*0.8), Batch_size=1000, crop_l=227)
    aug_val_images, aug_val_labels = augment_dataset_v2(raw_val_images, raw_val_labels, Dataset_size=val_size, Batch_size=1000, crop_l=227)

    print("raw dataset shape", raw_train_images.shape, raw_train_labels.shape, raw_val_images.shape, raw_val_labels.shape)
    print("Aug dataset shape", aug_train_images.shape, aug_train_labels.shape, aug_val_images.shape, aug_val_labels.shape)


    if data_aug:
        train_images, train_labels, val_images, val_labels = (aug_train_images, aug_train_labels, aug_val_images, aug_val_labels)
    else:
        train_images, train_labels, val_images, val_labels = (raw_train_images, raw_train_labels, raw_val_images, raw_val_labels)


    # define and train the model
    model = define_alexnet_keras(learning_rate)
    history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=training_epoch, validation_data=(val_images, val_labels))
    evaluation = model.evaluate(val_images, val_labels)
    print('loss: ', evaluation[0])
    print('accuracy', evaluation[1])
