import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import yaml


from model import Model


class Embedder:

    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device('cpu')

    def embed(self):
        print('Saving embeddings...')
        embeddings, vq_embeddings = [], []
        with torch.no_grad():
            for (data, _) in tqdm(self.dataloader):
                data = data.to(self.device)
                z = self.model._encoder(data)
                z = self.model._pre_vq_conv(z)
                _, quantized, _, _ = self.model._vq_vae(z)
                embeddings.append(z.cpu().numpy())
                vq_embeddings.append(quantized.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        vq_embeddings = np.concatenate(vq_embeddings, axis=0)
        print('Embeddings shape:', embeddings.shape)
        print('VQ Embeddings shape:', vq_embeddings.shape)
        results = {'pre_vq_embeddings': embeddings,
                   'vq_embeddings': vq_embeddings}
        torch.save(results, 'vq-vae-embeddings.pt')


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

    validation_data = datasets.CIFAR10(root=config['exp_params']['data_path'],
                                       train=False,
                                       download=False,
                                       transform=Compose([
                                           ToTensor(),
                                           Normalize(
                                               (0.5, 0.5, 0.5),
                                               (1.0, 1.0, 1.0)
                                           )
                                       ]))
    validation_loader = DataLoader(validation_data,
                            batch_size=256,
                            shuffle=False,
                            pin_memory=True)

    embedder = Embedder(model, validation_loader)
    embedder.embed()

if __name__ == '__main__':
    main()

