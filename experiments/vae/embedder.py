import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import yaml


from model import Model


def main():
    parser = argparse.ArgumentParser('Create embeddings from a trained model')
    parser.add_argument('--config', '-c', dest='filename', metavar='FILE', default='vae.yaml')
    parser.add_argument('--weights', '-w', default='model_state_dict.pt')
    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    device = torch.device('cpu')
    model = Model(**config['model_params']).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    train_data = datasets.CIFAR10(root=config['exp_params']['data_path'],
                                       train=True,
                                       download=False,
                                       transform=Compose([
                                           ToTensor(),
                                           Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                       ]))
    validation_data = datasets.CIFAR10(root=config['exp_params']['data_path'],
                                       train=False,
                                       download=False,
                                       transform=Compose([
                                           ToTensor(),
                                           Normalize((0.5, 0.5, 0.5),(1.0, 1.0, 1.0))
                                       ]))
    train_loader = DataLoader(train_data,
                              batch_size=256,
                              shuffle=False,
                              pin_memory=True)
    val_loader = DataLoader(validation_data,
                            batch_size=256,
                            shuffle=False,
                            pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Saving Train Embeddings...')
    embeddings, vq_embeddings = [], []
    with torch.no_grad():
        for (data, _) in tqdm(train_loader):
            data = data.to(device)
            z = model._encoder(data)
            z = model._pre_vq_conv(z)
            _, quantized, _, _ = model._vq_vae(z)
            embeddings.append(z.cpu().numpy())
            vq_embeddings.append(quantized.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    vq_embeddings = np.concatenate(vq_embeddings, axis=0)
    print('Train Embeddings shape:', type(embeddings), embeddings.shape)
    print('Train VQ Embeddings shape:', type(vq_embeddings), vq_embeddings.shape)
    print('Train Labels', type(train_data.targets), len(train_data.targets))
    results = {'pre_vq_embeddings': embeddings,
               'vq_embeddings': vq_embeddings,
               'targets': train_data.targets}
    torch.save(results, 'train-vq-vae-embeddings.pt')

    print('Saving Validation Embeddings')
    embeddings, vq_embeddings = [], []
    with torch.no_grad():
        for (data, _) in tqdm(val_loader):
            data = data.to(device)
            z = model._encoder(data)
            z = model._pre_vq_conv(z)
            _, quantized, _, _ = model._vq_vae(z)
            embeddings.append(z.cpu().numpy())
            vq_embeddings.append(quantized.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    vq_embeddings = np.concatenate(vq_embeddings, axis=0)
    print('Validation Embeddings shape:', type(embeddings), embeddings.shape)
    print('Validation VQ Embeddings shape:', type(vq_embeddings), vq_embeddings.shape)
    print('Validation Labels', type(validation_data.targets), len(validation_data.targets))
    results = {'pre_vq_embeddings': embeddings,
               'vq_embeddings': vq_embeddings,
               'targets': validation_data.targets}
    torch.save(results, 'val-vq-vae-embeddings.pt')



if __name__ == '__main__':
    main()

