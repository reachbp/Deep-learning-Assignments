__author__ = 'bharathipriyaa'

import torch, numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import data, argparse

parser = argparse.ArgumentParser(description='TSNE generator for Torch model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='model1',
                    help='path of the model file')
parser.add_argument('--title', type=str, default='model1',
                    help='Title of the plot')
args = parser.parse_args()

def loadmodel():
    model = torch.load(args.model)
    embed_matrix = model.encoder.weight
    embed_np_matrix = embed_matrix.data.numpy()
    corpus = data.Corpus(args.data)
    return embed_np_matrix[1:100], corpus.dictionary.idx2word

def saveTSNE(data ,  target):
    X_tsne = TSNE(learning_rate=100).fit_transform(data)
    print(X_tsne.shape)
    print(X_tsne)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X_tsne[:,0],X_tsne[:,1])
    for i, coord in enumerate(zip(X_tsne[:,0],X_tsne[:,1])):
        print(i, coord)
        ax.annotate(target[i], xy = coord )
    plt.title('TSNE plot for model {}'.format(args.title))
    plt.savefig('saved/tsne.png')
def main():
    data, target = loadmodel()
    print("Loaded torch model into numpy tensor")
    saveTSNE(data, target)
    print("TSNE saved as ")
main()

