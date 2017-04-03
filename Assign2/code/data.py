import os
import torch
import numpy as np
GLOVE_DIR = "../glove.6B"
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, pretrained = False, emsize = 50):


        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.pretrained = pretrained
        self.emsize = emsize
        self.embedmatrix = None
        if pretrained :
            self.embedmatrix = self.getEmbedding(self.dictionary, emsize)

    def getEmbedding(self, dictionary = None, emsize = 50):
        """
        :param dictionary: Dictionary which holds the mapping from id->word
        :param emsize: Size of the embedding
        :return: Embedding matrix of size len(dictionary)xemsize
        """
        embeddings_matrix = np.zeros((len(dictionary), emsize))
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            if word in dictionary.word2idx.keys():
                index = dictionary.word2idx[word]
                embeddings_matrix[index] = np.asarray(values[1:], dtype='float32')
        print("Generated the embeddings matrix ")
        print(embeddings_matrix)
        f.close()


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
