"""
query_v2.py contains utilities to calculate and return logits for trained models from
the pgdl challenge. Queries can be performed on models in the default pgdl format or on
custom models in Ani/Kiran's format.
"""
# External Libraries
import argparse
import json
import os
import tensorflow as tf
import tensorflow_datasets as tfds


# Custom Libraries
from models import VGG
from utils import model_def_to_keras_sequential
from utils import prepare_data, normalize


class QueryManager(object):
    def _init__(self):
        self._models_dir = os.path.join(os.getcwd(), 'data')
        self.custom_models_dir = os.path.join(self._models_dir, 'custom_models')
        self.pgdl_models_dir = os.path.join(self._models_dir, 'pgdl_models')

        self._custom_models = {}
        self._pgdl_models = {}
        self.cifar10_data_loaders = {}

    def prepare_cifar10(self, query_batch_size):
        if query_batch_size in self.cifar10_data_loaders:
            return self.cifar10_data_loaders[query_batch_size]
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = normalize(x_train, x_test)
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .map(prepare_data).shuffle(50000).batch(query_batch_size)
        self.cifar10_data_loaders[query_batch_size] = train_loader
        return train_loader

    def _load_pgdl(self, model_id):
        if model_id in self._pgdl_models:
            return self._pgdl_models[model_id]
        model_dir = os.path.join(self.pgdl_models_dir, model_id)
        if not os.path.exists(model_dir):
            raise OSError("Model directory does not exist!")
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        model = model_def_to_keras_sequential(config['model_config'])
        model.build([0] + config['input_shape'])
        weights_path = os.path.join(model_dir, 'weights.hdf5')
        initial_weights_path = os.path.join(model_dir, 'weights_init.hdf5')
        if os.path.exists(initial_weights_path):
            try:
                model.load_weights(initial_weights_path)
                model.initial_weights = model.get_weights()
            except ValueError as e:
                print('Error while loading initial weights of {} from {}'
                      .format(model_id, initial_weights_path))
                print(e)
        model.load_weights(weights_path)
        self._pgdl_models[model_id] = model
        return model

    def _load_custom(self, model_id):
        # return the model if it's already been constructed
        if model_id in self._custom_models:
            return self._custom_models[model_id]
        model_dir = os.path.join(self.custom_models_dir, model_id)
        if not os.path.exists(model_dir):
            raise OSError("Directory for custom model {} does not exist!"
                          .format(model_id))
        config_path = os.path.join(model_dir, 'config_{}'.format(model_id))
        weights_path = os.path.join(model_dir, 'weights_{}'.format(model_id))
        with open(config_path) as f:
            config = json.load(f)
        # print('Hyperparameters: ', config['hparams'], type(config['hparams']))
        # print('Train Time: {}'.format(config['train_time']))
        # train_acc, test_acc = config['train_acc'], config['test_acc']
        # print('Train Accuracy : {} | Test Accuracy {}'.format(train_acc, test_acc))
        hparams = argparse.Namespace(**config['hparams'])
        model = VGG(hparams)
        model.load_weights(weights_path)
        self._custom_models[model_id] = model
        return model

    def load_model(self, model_id, is_custom: bool):
        model = self._load_custom(model_id) if is_custom else self._load_pgdl(model_id)
        return model

    def query(self, model_id, query_batch_size, is_custom=False):
        if not 1 <= query_batch_size <= 50000:
            raise ValueError("Invalid batch size!")
        model = self.load_model(model_id, is_custom)
        data_loader = self.prepare_cifar10(query_batch_size)

        all_logits = []
        for idx, (data, target) in data_loader:
            @tf.function
            def step(image, label):
                output = model(image)
                # TODO: add other metrics
                return output
            logits = step(data, target)
            all_logits.append(logits)
        return all_logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PDGL Competition query utility.')
    parser.add_argument('model_id', type=str, nargs=1, help='Id of desired model for query')
    parser.add_argument('batch_size', type=int, nargs=1, help='Batch size to be used for query')
    parser.add_argument('--is_custom', type=bool, default=False, help='Specifies whether custom or competition model should be used')
    args = parser.parse_args()
    """
    Usage
    >>> manager = QueryManager()
    >>> manager.query(42, 128, custom=True)
    [...] # returns list of all the logits if batch size is 128 then this is a list of len 391
    """
