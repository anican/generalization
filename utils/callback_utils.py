"""
Callback utilities.
"""
from argparse import Namespace


def get_experiment_str(args: Namespace) -> str:
    """
    Returns concatenated string of hyperparameters associated with a certain experiment
    setting.

    :param args: hyperparameters for the given experiment
    :return:
    """
    batch_size = args.batch_size
    conv_width = args.conv_width
    dropout = args.dropout
    num_dense = args.num_dense
    num_block = args.num_block
    weight_decay = args.weight_decay
    template = 'convwidth{}_nblocks{}_ndense{}_batchsize{}_dpout{}_wgtdec{}'
    return template.format(conv_width, num_block, num_dense, batch_size, dropout,
                           weight_decay)


