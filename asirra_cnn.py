from functions_used import load_cifar_10, define_model_cifar10, train_and_evaluate_model, summarize_diagnostics,\
    define_alexnet_keras, random_crop_reflect, center_crop
from dataset import asirra as dataset
import tensorflow as tf
import numpy as np
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import AccuracyEvaluator as Evaluator

import os

if __name__ == '__main__':

    # Parameter Setting
    learning_rate = 0.01
    training_epoch = 10
    batch_size = 32
    display_step = 20

    root_dir = os.path.join("C:/Users", "Chan", "asirra")
    trainval_dir = os.path.join(root_dir, "train")

    print(tf.__version__)
    # with Keras

    trainval_images, trainval_labels = dataset.read_asirra_subset(trainval_dir, one_hot=True)

    trainval_size = trainval_images.shape[0]
    val_size = int(trainval_size * 0.1)

    raw_images, train_labels = trainval_images[val_size:], trainval_labels[val_size:]
    # train_images = center_crop(raw_images, 227)
    train_images = random_crop_reflect(raw_images, 227)
    # train_images = np.append(train_images, crop_images)
    # print(train_images.shape)
    # train_images = random_crop_reflect(train_images, 227)
    val_images, val_labels = trainval_images[:val_size], trainval_labels[:val_size]
    val_images = random_crop_reflect(val_images, 227)
    # val_set = dataset.DataSet()
    # train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])
    print(train_images.shape, train_labels.shape, val_images.shape, val_labels.shape)
    model = define_alexnet_keras(learning_rate)
    history = train_and_evaluate_model(model, train_images, train_labels,
                                       val_images, val_labels, batch_size, training_epoch)
    summarize_diagnostics(history)


    # without keras
    '''
    X_trainval, y_trainval = dataset.read_asirra_subset(trainval_dir, one_hot=True)
    trainval_size = X_trainval.shape[0]
    val_size = int(trainval_size * 0.2)  # FIXME
    val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
    train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

    hp_d = dict()
    image_mean = train_set.images.mean(axis=(0, 1, 2))  # mean image
    hp_d['image_mean'] = image_mean

    # FIXME: Training hyperparameters
    hp_d['batch_size'] = 256
    hp_d['num_epochs'] = 300

    hp_d['augment_train'] = True
    hp_d['augment_pred'] = True

    hp_d['init_learning_rate'] = 0.01
    hp_d['momentum'] = 0.9
    hp_d['learning_rate_patience'] = 30
    hp_d['learning_rate_decay'] = 0.1
    hp_d['eps'] = 1e-8

    # FIXME: Regularization hyperparameters
    hp_d['weight_decay'] = 0.0005
    hp_d['dropout_prob'] = 0.5

    # FIXME: Evaluation hyperparameters
    hp_d['score_threshold'] = 1e-4
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model = ConvNet([227, 227, 3], 2, **hp_d)
    evaluator = Evaluator()
    optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

    sess = tf.Session(graph=graph, config=config)
    train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)
    '''