#!/usr/bin/env python
"""
The query utility is a python function and CLI program

Input: model configuration of a VGG-like network (in the Keras H5 format) and
       batch of cifar10 data.
Output: model and logits


Example:
>>>
"""
import argparse
from argparse import Namespace
import json
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import tqdm


from models import VGG
from utils import prepare_data, normalize


@tf.function
def step(data, target):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    logits = model(data, training=False)
    accuracy(target, logits)


def validate_weights(f):
    if not (f.endswith('h5') or f.endswith('hdf5')):
        raise argparse.ArgumentParser('Invalid filetype!')
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError('{0} does not exist!'.format(f))
    return f


def validate_config(f):
    if not (f.endswith('json')):
        raise argparse.ArgumentParser('Invalid filetype!')
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError('{0} does not exist!'.format(f))
    return f


def construct_model(config_path, weights_path):
    # Recover JSON contents
    with open(config_path) as config_file:
        config = json.load(config_file)
        print('Hyperparameters: ', config['hparams'], type(config['hparams']))
        print('Train Time: {}'.format(config['train_time']))
        train_acc, test_acc = config['train_acc'], config['test_acc']
        print('Train Accuracy : {} | Test Accuracy {}'.format(train_acc, test_acc))
        args = Namespace(**config['hparams'])
    model = VGG(args)
    model.load_weights(weights_path)
    return model


def query(model, dataset, batch_size):
    start = time.time()
    data = next(iter(dataset.batch(batch_size)))['image']
    logits = model(data)
    end = time.time()
    print("Query took: {}\n", end - start)
    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser('~~Query VGG Net from the command line~~')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('-c', '--config', dest='config', type=validate_config,
                        required=True, help='Specify a valid JSON config file, which '
                                            'corresponds to the chosen weight file.')
    parser.add_argument('-w', '--weights', dest='weights', type=validate_weights,
                        required=True, help='Specify a valid H5 weights file, which '
                                            'corresponds to the chosen config file.')
    args = parser.parse_args()
    print('Data Loading...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = normalize(x_train, x_test)
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .map(prepare_data).shuffle(50000).batch(args.batch_size)
    # test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    #     .map(prepare_data).shuffle(10000).batch(args.batch_size)

    model = construct_model(args.config, args.weights)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    for (data, target) in tqdm.tqdm(train_loader):
        step(data=data, target=target)
    print(accuracy.result().numpy() * 100)

    #### TFDS METHOD ####
    # dataset = tfds.load('cifar10', split='train', shuffle_files=True)
    #
    # # sample = next(iter(dataset.batch(128)))['image']
    # accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    # model = construct_model(args.config, args.weights)
    #
    # # print((list(dataset)[0]['image'].shape))
    # for example in tqdm.tqdm(tfds.as_numpy(dataset)):
    #     image, label = example['image'], example['label']
    #     step(data=image, target=label)
    # print(accuracy.result().numpy() * 100)



