import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import yaml


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


import torchvision.datasets as datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.utils import make_grid


from experiments.vae.model import Model


def train():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c', dest="filename", metavar='FILE',
                        help='path to the config file', default='vae.yaml')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)

    use_cuda = torch.cuda.is_available()
    gpu_idx = config['exp_params']['gpu']
    device = torch.device('cuda:{}'.format(gpu_idx) if use_cuda else 'cpu')
    print('Using Device:', device)
    training_data = datasets.CIFAR10(root=config['exp_params']['data_path'],
                                     train=True,
                                     download=True,
                                     transform=Compose([
                                         ToTensor(),
                                         Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    validation_data = datasets.CIFAR10(root=config['exp_params']['data_path'],
                                       train=False,
                                       download=True,
                                       transform=Compose([
                                           ToTensor(),
                                           Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                       ]))
    train_loader = DataLoader(training_data,
                              batch_size=config['exp_params']['batch_size'],
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(validation_data,
                            batch_size=32,
                            shuffle=True,
                            pin_memory=True)
    data_variance = np.var(training_data.data / 255.0)
    model = Model(**config['model_params']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['exp_params']['lr'])

    model.train()
    train_recon_errors, train_perplexities = [], []

    for step in range(config['exp_params']['train_steps']):
        # sampling step
        (data, _) = next(iter(train_loader))
        data = data.to(device)
        optimizer.zero_grad()
        # train step
        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        # update step
        loss.backward()
        optimizer.step()

        train_recon_errors.append(recon_error.item())
        train_perplexities.append(perplexity.item())

        if (step + 1) % config['exp_params']['print_interval'] == 0:
            print('%d iterations' % (step + 1))
            print('recon_error: %.3f' % np.mean(train_recon_errors[-100:]))
            print('perplexity: %.3f\n' % np.mean(train_perplexities[-100:]))
    train_recon_error_smooth = savgol_filter(train_recon_errors, 201, 7)
    train_perplexities_smooth = savgol_filter(train_perplexities, 201, 7)
    train_results = {'errors': train_recon_error_smooth,
                     'perplexities': train_perplexities_smooth}
    torch.save(train_results, 'train_results.pt')
    torch.save(model.state_dict(), 'model_state_dict.pt')


if __name__ == '__main__':
    train()
