__author__ = 'bharathipriyaa'

import torch, numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import data, argparse, pickle 

parser = argparse.ArgumentParser(description='TSNE generator for Torch model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='saved/trial',
                    help='path of the model file')
parser.add_argument('--title', type=str, default='model1',
                    help='Title of the plot')
args = parser.parse_args()

def loadmodel():
    model = torch.load(args.model)
    embed_matrix = model.encoder.weight
    embed_np_matrix = embed_matrix.data.cpu().numpy()
    idx2word = pickle.load(open('idx2wordpenn.p', 'rb')) 
    print("Word to index diction")
    print(idx2word)
    return embed_np_matrix, idx2word

def saveTSNE(data ,  target):
    X_tsne = TSNE(learning_rate=500, n_iter= 5000, verbose=  1).fit_transform(data)
    pickle.dump(X_tsne, open('tsne/tsnemat'+args.title+'.p', 'wb'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X_tsne[:,0],X_tsne[:,1])
    for i, coord in enumerate(zip(X_tsne[:,0],X_tsne[:,1])):
         ax.annotate(target[i], xy = coord )
    plt.title('TSNE plot for model {}'.format(args.title))
    plt.savefig('tsne/'+args.title+'.png')
def main():
    data, target = loadmodel()
    print("Loaded torch model into numpy tensor")
    saveTSNE(data, target)
    print("TSNE saved as "+ args.title)
    print("TSNE matrix saved as tsne/tsne"+args.title+".p")
main()

