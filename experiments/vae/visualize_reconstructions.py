import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.utils import make_grid


from model import Model


def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c', dest="filename", metavar='FILE',
                        help='path to the config file', default='vae.yaml')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(**config['model_params'])
    model.load_state_dict(torch.load('model_state_dict.pt',
                          map_location=torch.device('cpu')))
    model.eval()
    validation_data = datasets.CIFAR10(root=config['exp_params']['data_path'],
                                       train=False,
                                       download=True,
                                       transform=Compose([
                                           ToTensor(),
                                           Normalize(
                                               (0.5, 0.5, 0.5),
                                               (1.0, 1.0, 1.0)
                                           )
                                       ]))
    validation_loader = DataLoader(validation_data,
                            batch_size=32,
                            shuffle=True,
                            pin_memory=True)
    (valid_originals, _) = next(iter(validation_loader))
    valid_originals = valid_originals.to(device)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    show(make_grid(valid_reconstructions.cpu().data)+0.5, )
    # show(make_grid(valid_originals.cpu()+0.5))
    plt.show()

if __name__ == '__main__':
    main()
