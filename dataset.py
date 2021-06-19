
# 1- 
# Wikipedia size is 12 GB. You can download chunks of size 100x MB, still need to do parsing as it is XML format.



# 2- Alternative is the brown Corpus - comes with nltk 
# nltk.download('brown')
from nltk.corpus import brown
def get_sentences():
  # returns 57340 of the Brown corpus (around 57340 sentences) - much smaller than Wikipedia but OK for the course 
  # each sentence is represented as a list of individual string tokens
  return brown.sents()      # list of lists, each inner list is a sentence = list of words already tokenized 


# 3- Text8Corpus from Gensim
gensim.models.word2vec.Text8Corpus(fname, max_sentence_length=10000) # Iterate over sentences from the “text8” corpus, unzipped from http://mattmahoney.net/dc/text8.zip.
# see also LineSentence in word2vec fromm Gensim



# 4- # textmining datasets   - It is used for text classifier bow_classifier.py
""" https://github.com/Cynwell/Text-Level-GNN """
# open the file.txt, then see raw data -> wget url


# 5- glove word embedding 
# https://nlp.stanford.edu/projects/glove/
# https://www.kaggle.com/watts2/glove6b50dtxt

# 6- Binary file for word2vec 
brew install wget
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d GoogleNews-vectors-negative300.bin.gz
from gensim import models
w = models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)


# 6- dataset for sentiment analysis 
from keras.datasets import imdb

# 7- https://gutenberg.org/: 
# dataset for word embedding like word2vec basic model - go to https://gutenberg.org/ebooks/8172 - download UTF-8 format 
# ALSO, https://www.gutenberg.org/cache/epub/10773/pg10773.txt is used for text generations by letters

  




