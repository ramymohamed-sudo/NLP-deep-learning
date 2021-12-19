
# 1- 
# Wikipedia size is 12 GB. You can download chunks of size 100x MB, still need to do parsing as it is XML format.



# 2- Alternative is the brown Corpus - comes with nltk 
# nltk.download('brown')
from nltk.corpus import brown
def get_sentences():
  # returns 57340 of the Brown corpus (around 57340 sentences) - much smaller than Wikipedia but OK for the course 
  # each sentence is represented as a list of individual string tokens
  return brown.sents()      # list of lists, each inner list is a sentence = list of words already tokenized 
# it has categories (unlike gutenberg which has many files) 
[['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 
  'religion', 'reviews', 'romance', 'science_fiction']] - textual data under each category 


# 3- Text8Corpus from Gensim
gensim.models.word2vec.Text8Corpus(fname, max_sentence_length=10000) # Iterate over sentences from the “text8” corpus, unzipped from http://mattmahoney.net/dc/text8.zip.
# see also LineSentence in word2vec fromm Gensim


# 4- Binary file for WORD2VEC 
brew install wget
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d GoogleNews-vectors-negative300.bin.gz
from gensim import models
w = models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)


# 5- GLOVE *.txt word embedding files
# https://nlp.stanford.edu/projects/glove/
# 50d, 100d, 200d, 300d -> after unzipping the files -> glove6B-300d.txt
# https://www.kaggle.com/watts2/glove6b50dtxt
# in the file each line word [vector]
path = 'glove.6B.50d.txt'
embed_index = {}
with open(path, encoding='utf8') as f:
    lines = f.readline()
    for line in lines:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:])
        embed_index[word] = coeffs


# 6- # textmining datasets   - It is used for text classifier bow_classifier.py
""" https://github.com/Cynwell/Text-Level-GNN """
# open the file.txt, then see raw data -> wget url

# 7- dataset for sentiment analysis 
from keras.datasets import imdb

# 8- Yelp dataset for sentiment analysis
path = './DEEP_NLP_resources 2/data/yelp.csv'
df = pd.read_csv(path)


# 9- https://gutenberg.org/: 
# dataset for word embedding like word2vec basic model - go to https://gutenberg.org/ebooks/8172 - download UTF-8 format 
# ALSO, https://www.gutenberg.org/cache/epub/10773/pg10773.txt is used for text generations by letters

# 10- gutenberg from nltk 
nltk.download('gutenberg')
print(gutenberg.fileids())    # I used 'bible-kjv.txt'  

11- dataset for sentiment analysis
http://help.sentiment140.com/for-students/
  stanford link

12- dataset used for wordembedding with gensim - from Kaggle
https://www.kaggle.com/rootuser/worldnews-on-reddit
the data is .csv file
df = pd.read_csv('') # title column for the embedding model

13- googles-trained-word2vec-model from Kaggle
https://www.kaggle.com/umbertogriffo/googles-trained-word2vec-model-in-python
# .bin file(s)

14- Airline Travel Information System (ATIS), ATIS is a well-known dataset for intent classification.
wget https://github.com/PacktPublishing/Mastering-spaCy/blob/main/Chapter06/data/atis_intents.csv
  
15- Toxic_data from Kaggle for binary classification - multi-label data - insult/thread/obscence/identity_hate/severe_toxic
# Download the data:
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# Download the word vectors:
# http://nlp.stanford.edu/data/glove.6B.zip




