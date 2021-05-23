# Glove and word2vec basic codes from here 
# https://ibm-learning.udemy.com/course/natural-language-processing-with-deep-learning-in-python/learn/lecture/10770690#overview

# For most of people, vocab size = 20,000 

# 1- Word2vec - vocab size = 3 million

# 1-0- Basic code from the udemy course


# 1-1- from Gensim
import gensim
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('path/file.bin', binary=True)    # # 3 millions words and phrases 



from gensim.models import Word2Vec
w2v_model = Word2Vec(brown.sents(), size=128, window=5, min_count=3, workers=4)     # word2vec = Word2Vec(all_words, min_count=2)
ger_vec = w2v_model.wv['Germany']
w2v_model.wv.most_similar('Vienna')
w2v_model.wv.most_similar(positive=['woman',  'king'], negative=['man'],topn=5)
better_w2v_model = Word2Vec(Text8Corpus('data_text8_corpus.txt'), size=100, window=5, min_count=150, workers=4)
words_voc = []
for word in better_w2v_model.wv.vocab:
      words_voc.append(better_w2v_model.wv[word])
    
    
    
    
# gensim in general
# gensim lemmatizer 
# wordnet lemmatizer


# 1-2- from Spacy 


# 2- Glove    vocab size = 400,000





# 3- Word2vec using PMI implementation 
# long time to download - comes in a binary format - needs to parse it 
# this parsing code already written in gensim library 
