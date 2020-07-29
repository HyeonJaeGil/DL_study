from functions_used import load_cifar_10, define_model_cifar10, train_and_evaluate_model, summarize_diagnostics,\
    define_alexnet_keras, define_alexnet_keras_rev, random_crop_reflect, center_crop, corner_center_crop_reflect, augment_dataset
from dataset import asirra as dataset
import tensorflow as tf
import numpy as np
# from learning.optimizers import MomentumOptimizer as Optimizer
# from learning.evaluators import AccuracyEvaluator as Evaluator

import os

if __name__ == '__main__':

    # Parameter Setting
    learning_rate = 0.01
    training_epoch = 10
    batch_size = 32
    display_step = 20
    data_aug = True

    root_dir = os.path.join("C:/Users", "Chan", "asirra")
    trainval_dir = os.path.join(root_dir, "debug")  # change train to debug for checking just structure of code

    print(trainval_dir)

    print(tf.__version__)
    # with Keras

    # load the dataset
    trainval_images, trainval_labels = dataset.read_asirra_subset(trainval_dir, one_hot=True)

    trainval_size = trainval_images.shape[0]
    val_size = int(trainval_size * 0.2)

    raw_train_images, raw_train_labels = trainval_images[val_size:], trainval_labels[val_size:]
    raw_val_images, raw_val_labels = trainval_images[:val_size], trainval_labels[:val_size]

    # augument the dataset(12 times bigger)
    aug_train_images, aug_train_labels = augment_dataset(raw_train_images, raw_train_labels, crop_l=227)
    aug_val_images, aug_val_labels = augment_dataset(raw_val_images, raw_val_labels, crop_l=227)

    print("raw dataset shape", raw_train_images.shape, raw_train_labels.shape, raw_val_images.shape, raw_val_labels.shape)
    print("Aug dataset shape", aug_train_images.shape, aug_train_labels.shape, aug_val_images.shape, aug_val_labels.shape)


    if data_aug:
        train_images, train_labels, val_images, val_labels = (aug_train_images, aug_train_labels, aug_val_images, aug_val_labels)
    else:
        train_images, train_labels, val_images, val_labels = (raw_train_images, raw_train_labels, raw_val_images, raw_val_labels)


    # define and train the model
    model = define_alexnet_keras_rev(learning_rate)
    history = train_and_evaluate_model(model, train_images, train_labels,
                                       val_images, val_labels, batch_size, training_epoch)
    summarize_diagnostics(history)