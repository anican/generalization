import matplotlib.pyplot as plt
import os
import torch


if __name__ == '__main__':
    PATH = os.path.join(os.getcwd(), 'train_results.pt')
    results = torch.load(PATH)

    train_error = results['errors']
    train_perplexities = results['perplexities']

    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,2,1)
    ax.plot(train_error)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1,2,2)
    ax.plot(train_perplexities)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')
    plt.show()
