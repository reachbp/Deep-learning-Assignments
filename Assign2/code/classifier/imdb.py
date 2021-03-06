__author__ = 'bharathipriyaa'


import os, pprint, json
import pickle, torch
import numpy as np
import re, argparse
import random
# Training settings
parser = argparse.ArgumentParser(description='Processing IMDB Dataset ')
parser.add_argument('--filename', type=str, default='dummy.json',
                    help='filename  ')

args = parser.parse_args()
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = ["null"]
        self.word2count = {}
	
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word2count[word] = 0
            self.word2count[word] = self.word2count[word] + 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class ImdbDataset(object):

    def __init__(self,filename):
        self.filename = filename
        self.dictionary = Dictionary()
        self.data = []
        self.target = []

    def loadDataset(self, filename):
        assert os.path.exists(filename)
        for line in open(filename, 'r'):
            self.processReviewText(line)

        print("Size in the vocabulary ",len(self.dictionary.word2idx.keys()) )
        # Prune words with frequency of 5 or less
        """
        for word in self.dictionary.word2count.keys():
            if  self.dictionary.word2count[word] < 5:
                index_to_remove = self.dictionary.word2idx[word]
                last_index = self.dictionary.__len__() - 1
                last_word = self.dictionary.idx2word[last_index]
                self.dictionary.idx2word[index_to_remove] = last_word 
                self.dictionary.word2idx[last_word] = index_to_remove
                self.dictionary.word2idx[word] = 0
                self.dictionary.idx2word.pop()
		"""
        print("Size of vocabulary before & after pruning low-freq words ",len(self.dictionary.word2idx.keys()), self.dictionary.__len__() )
        for line in open(filename, 'r'):
            line = line.split(',')
            review = ','.join(line[1:])
            #print(review)
            self.data.append(self.tokenizeReviewText(review))
            if "pos" in line[0]:
                self.target.append(0)
            else:
                self.target.append(1)
	    
        print("============== All reviews read into list ==============")


        pickle.dump(self.data, open('data_imdb_test.pkl', 'wb'))
        pickle.dump(self.target, open('target_imdb_test.pkl', 'wb'))
        #pickle.dump(self.dictionary.idx2word, open('idx2word.pkl', 'wb'))
        #print("Words in the vocabulary ",self.dictionary.word2idx )
        #pickle.dump(self.dictionary.word2idx, open('vocab_imdb.p', 'wb'))
        print("Data saved to pickle file")

    def processReviewText(self, text):
        text = text.lower()
        text = re.sub('[^A-Za-z0-9\s]+', ' ', text).lower()
        for word in text.split():
            self.dictionary.add_word(word)


    def tokenizeReviewText(self, text):
        text = re.sub('[^A-Za-z0-9\s]+', ' ', text).lower()
        text = text.lower()
        ids = np.zeros(50)
        token = 0
        for word in text.split():
            if token >= 50:
                break;
            ids[token] = self.dictionary.word2idx[word]
            token += 1
        return ids

class ReviewPair(object):
    def __init__(self, text, score):
        self.text = text
        self.score = score


def main():
    filename = args.filename
    dataset = ImdbDataset(filename)
    dataset.loadDataset(filename)
main()
