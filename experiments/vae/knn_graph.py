import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3)
    args = parser.parse_args()
    print('Loading training data...')
    metadata = load('train-vq-vae-embeddings.pt')
    data_train = metadata['vq_embeddings'].reshape(-1, 64*8*8)
    print('X:', data_train.shape)
    targets_train = metadata['targets']
    print('y:', len(targets_train))

    print('Fitting classifier to training data...')
    clf = KNeighborsClassifier(n_neighbors=args.k)
    clf.fit(data_train, targets_train)

    train_score = clf.score(data_train, targets_train)
    print('Train Accuracy:', train_score)
    print('\nLoading validation data...')
    metadata = load('val-vq-vae-embeddings.pt')
    data_val = metadata['vq_embeddings'].reshape(-1, 64*8*8)
    print('X_val:', data_val.shape)
    targets_val = metadata['targets']
    val_score = clf.score(data_val, targets_val)




if __name__ == '__main__':
    main()

