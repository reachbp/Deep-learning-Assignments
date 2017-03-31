__author__ = 'bharathipriyaa'


import os, path, json, pprint
import pickle, torch
import numpy as np
import re

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


class YelpDataset(object):

    def __init__(self,filename):
        self.filename = filename
        self.dictionary = Dictionary()
        self.data = []
        self.target = []

    def loadDataset(self, filename):
        assert os.path.exists(filename)
        for line in open(filename, 'r'):
            review = json.loads(line)
            self.processReviewText(review['text'])

        for line in open(filename, 'r'):
            review = json.loads(line)
            self.data.append(self.tokenizeReviewText(review['text']))
            self.target.append(review['stars']-1)

        print("============== All reviews read into list ==============")

        print("Number of reviews loaded ", len(self.data), len(self.target))
        pickle.dump(self.data, open('data.pkl', 'wb'))
        pickle.dump(self.target, open('target.pkl', 'wb'))
        print("Words in the vocabulary ",self.dictionary.word2idx )
        pickle.dump(self.dictionary.word2idx, open('vocab.p', 'wb'))
        print("Data saved to pickle file")

    def processReviewText(self, text):
        text = re.sub('[^A-Za-z0-9\s]+', ' ', text).lower()
        for word in text.split():
            self.dictionary.add_word(word)

    def tokenizeReviewText(self, text):
        text = re.sub('[^A-Za-z0-9\s]+', ' ', text).lower()
        numWords = len(text.split())
        ids = np.zeros(numWords+1)
        token = 0
        for word in text.split():
            ids[token] = self.dictionary.word2idx[word]
            token += 1
        return ids[0:50]

class ReviewPair(object):
    def __init__(self, text, score):
        self.text = text
        self.score = score


def main():
    filename = "../data/sentiment/yelp_training_set_review.json"
    dataset = YelpDataset(filename)
    dataset.loadDataset(filename)

main()
