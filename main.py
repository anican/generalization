import argparse
import os
import tensorflow as tf
import time
from tqdm import tqdm


from models import VGG, Network
from utils import prepare_data, normalize, get_experiment_str


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
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_block', type=int, default=6)
    parser.add_argument('--num_dense', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    hparams = parser.parse_args()

    device_name = tf.test.gpu_device_name()
    print('Using', 'gpu' if hparams.cuda else 'cpu')
    if hparams.cuda:
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    tf.random.set_seed(hparams.seed)
    print('loading data...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = normalize(x_train, x_test)
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(prepare_data).shuffle(50000).batch(hparams.batch_size)
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(prepare_data).shuffle(10000).batch(hparams.batch_size)
    print('finished loading data...')

    model = Network() # VGG(hparams)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    log_dir = os.getcwd() + '/logs/gradient_tape/' + get_experiment_str(hparams)
    print("logging in ...", log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)
    # TODO: specify dtypes for metrics?
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # val_loss = tf.keras.metrics.Mean(name='val_loss')
    # val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    print('training model...')
    start = time.time()
    for epoch in tqdm(range(hparams.epochs)):
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
        print(template.format(epoch + 1, train_loss.result(),
                              train_accuracy.result() * 100))
    end = time.time()
    # save_dir = os.getcwd() + '/checkpoints/' + get_experiment_str(hparams) + '.h5'
    # print('saving model...', save_dir)
    # model.save(save_dir)
    print('Test Accuracy: {}, Training Time: {}'.format(test_accuracy.result() * 100,
                                                        end - start))
