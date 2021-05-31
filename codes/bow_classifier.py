
# some examples of bow are Word counting and TFIDF 
# example from sklearn 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
BOW  = X.toarray()
vectorizer.get_feature_names()
bow = pd.DataFrame(BOW,columns=vectorizer.get_feature_names())

# For the source code of bag of words 
# https://www.freecodecamp.org/news/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04/
# bow can be seen as a special case of n-gram where n=1
# CBOW = extension of bigram 

# word2vec trains words against other words that neighbor them. To do so, it uses CBOW method, or the skip-gram method. 
# Skip-gram uses auto-encoder input projection 
# Problem with BOW and TFIDF, they do not store the semantic information. 



# word analogy and word similarity to gudge on feature vector 




""" Below is the code to use bow for pretrained vectors based on Glove and Word2vec  """ 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from gensim.models import KeyedVectors

# data are tab separated
train = pd.read_csv('./r8-train-all-terms.txt', sep='\t', header=None)
test = pd.read_csv('./r8-test-all-terms.txt', sep='\t', header=None)
train.columns = ['label', 'content']
test.columns = ['label', 'content']
print(train.head())
# print(train.groupby('label').size())
# print(train[train['label'] == 'grain'].count())
# print(train.info())
# print(" ....", train.label.unique())
# print(train.describe())


# The 2 classes below act as sklearn data transform
# each has 3 functions, fit, transform, and fir_transform
class GloveVectorizer():
    def __init__(self) -> None:
        print('loading word vectors....')
        word2vec = {}
        embedding = []
        idx2word = []

        with open('./glove.6B.50d.txt') as f:
            # it is a space sperated each line has # word vector
            for line in f:
                values = line.split()  # default delimiter = whitespace
                word = values[0]
                vec = np.asfarray(values[1:])
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)
        print(f'Found {len(word2vec)} word vectors')

        # save for later
        self.word2vec = word2vec
        self.embedding = np.asfarray(embedding)
        self.idx2word = idx2word
        # print("type(self.embedding)", type(self.embedding))
        # print(type(self.embedding[0]), self.embedding[0].shape)
        self.V, self.D = self.embedding.shape
        # print("V and D", self.V, self.D)

    def fit(self, data):
        pass

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        print("data: ", data)
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print(f"number of samples with no words found is {emptycount} / {len(data)}")
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# no lower case in word2vec as it has both upper and lower case words

class Word2VecVectorizer():
    def __init__(self) -> None:
        print('loading in word vectors....')
        self.word_vectors = KeyedVectors.load_word2vec_format(
            './GoogleNews-vectors-negative300.bin', binary=True
        )
        print('Finished loading word vectors....')

    def fit(self, data):
        pass

    def transform(self, data):
        # determine the dimensionality of vectors
        self.D = self.word_vectors['king'].shape[0]
        print("self.D", self.D)
        #  self.D = v.shape[0]
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.split()   # line.split('\t')) default delimiter
            vecs = []
            m = 0
            for word in tokens:
                try:
                    vec = self.word_vectors[word]
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass

            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print(f"number of samples with no words found is {emptycount} / {len(data)}")
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# vectorizer = GloveVectorizer()
vectorizer = Word2VecVectorizer()
Xtrain = vectorizer.fit_transform(train.content)
print(Xtrain.shape)
Ytrain = train.label

Xtest = vectorizer.fit_transform(test.content)
print(Xtest.shape)
Ytest = test.label


# Create the classifer model
model = RandomForestClassifier(n_estimators=10, n_jobs=-1)   # was 50
model.fit(Xtrain, Ytrain)
print(f'Train score: {model.score(Xtrain, Ytrain)}')
print(f'Test score: {model.score(Xtest, Ytest)}')

""" This score measures how many labels the model got right out of the total number of predictions. 
You can think of this as the percent of predictions that were correct. 
This is super easy to calculate with Scikit-Learn using the true labels from the test set 
and the predicted labels for the test set."""



