import argparse
import json
import os
import tensorflow as tf
import time


from models import VGG, Network
from utils import prepare_data, normalize


@tf.function
def train_step(data, target):
    """
    This function, given a minibatch of data, computes a forward pass over the model and
    then optimizes the model with a backward pass update over the network.
    :param data:
    :param target:
    :return:
    """
    with tf.GradientTape() as tape:
        # use Gradient tape to train the model
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(data, training=True)
        loss = criterion(target, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(target, logits)


@tf.function
def test_step(data, target):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    logits = model(data, training=False)
    loss = criterion(target, logits)
    test_loss(loss)
    test_accuracy(target, logits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--conv_width', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.01)
    # TODO: change to 500
    parser.add_argument('--gpu_idx', type=int)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--model_num', type=int)
    parser.add_argument('--num_block', type=int, default=6)
    parser.add_argument('--num_dense', type=int, default=2)
    parser.add_argument('--seed', type=int, default=446)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    hparams = parser.parse_args()

    if tf.test.is_gpu_available():
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[hparams.gpu_idx], 'GPU')

    tf.random.set_seed(hparams.seed)
    print('Data Loading...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = normalize(x_train, x_test)
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .map(prepare_data).shuffle(50000).batch(hparams.batch_size)
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
        .map(prepare_data).shuffle(10000).batch(hparams.batch_size)
    print('End Data Loading...\n')

    model = Network() # VGG(hparams)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    log_dir = os.getcwd() + '/logs/gradient_tape/' + 'model_{}'.format(hparams.model_num)
    print("Logging...\n", log_dir, '\n')
    summary_writer = tf.summary.create_file_writer(log_dir)
    # TODO: specify dtypes for metrics?
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # val_loss = tf.keras.metrics.Mean(name='val_loss')
    # val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    print('Training Model...')
    start = time.time()
    epoch = 0
    while True:
        train_loss.reset_states()
        train_accuracy.reset_states()

        for idx, (data, target) in enumerate(train_loader):
            train_step(data, target)
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            tf.summary.scalar('train_acc', train_accuracy.result(), step=epoch)

        for idx, (data, target) in enumerate(test_loader):
            test_step(data, target)
        with summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
            tf.summary.scalar('test_acc', test_accuracy.result(), step=epoch)
        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        loss_val = train_loss.result()
        acc_val = train_accuracy.result() * 100
        print(template.format(epoch + 1, loss_val, acc_val), '\n')
        epoch += 1
        if loss_val < hparams.epsilon or epoch > hparams.max_epochs:
            # if we've reached the cross-entropy criterion stopping point
            # or if we've exceeded the number of permissible epochs
            break
    end = time.time()
    total_time = end - start

    print('Stopping criterion reached!\nSaving model...')
    h5_path = os.path.join(os.getcwd(), 'weights_{}.h5'.format(hparams.model_num))
    model.save_weights(h5_path)
    info = {}
    train_acc_val = train_accuracy.result() * 100
    test_acc_val = test_accuracy.result() * 100
    info['hparams'] = vars(hparams)
    info['train_acc'] = float(train_acc_val.numpy())
    info['test_acc'] = float(test_acc_val.numpy())
    info['train_time'] = total_time
    json_path = os.path.join(os.getcwd(), 'config_{}.json'.format(hparams.model_num))
    with open(json_path, 'w') as file:
        json.dump(info, file)
    with summary_writer.as_default():
        tf.summary.scalar('Train Time (sec)', total_time, step=0)
    print('Training Time Elapsed: {} | Training Accuracy: {}'.format(total_time, train_acc_val))
    print('Test Accuracy: {}'.format(test_acc_val, '\n'))
    print('done')
"""
TODO: 
- saving in H5 format specifically (sequential giving some issues)
- checking functionality for (multi)-gpu
"""

