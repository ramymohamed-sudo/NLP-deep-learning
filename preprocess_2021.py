
# cleaning punctuation - stop words - word/sent tokenize and lemmatize
from string import punctuation
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
import unicodedata

porter = PorterStemmer()
lancaster = LancasterStemmer()
stemmed = [porter.stem(word), lancaster.stem(word) for word in list_of_tokens]

import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

url = 'https://raw.githubusercontent.com/laxmimerit/twitter-disaster-prediction-dataset/master/train.csv'
tweeter = pd.read_csv(url)
tweeter.head()

!pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall
import preprocess_kgptalkie as kgp
tweet = kgp.get_basic_features(tweeter)

def get_clean(x):
  x = str(x).lower().replace('\\','').replace('_',' ').replace('.',' ')         # "I like this movie ...".lower
  x = kgp.cont_exp(x)
  x = kgp.remove_emails(x)
  x = kgp.remove_urls(x)
  x = kgp.remove_html_tags(x)
  x = kgp.remove_rt(x)
  x = kgp.remove_accented_chars(x)
  x = kgp.remove_special_chars(x)
  x = kgp.remove_dups_char(x)
  return x




def correct_word_miss_spelling():
  pass
tweet['text'] = tweet['text'].apply(lambda x: get_clean(x))




def preprocessing(text):
    words = word_tokenize(text)
    tokens = [w for w in words if w.lower() not in remove_terms]
    # you can also try to apply the commented conditions below
    # stopw = stopwords.word('english')   # stopw like I, my, you, myslelf 
    # tokens = [token for token in tokens if token not in stopw]
    # remove words less than 3 words
    # tokens = [word for word in tokens if len(word) >=3]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]  #[lemma.lemmatize(word, wordnet.VERB) for word in tokens] to tell lemmatizer words are verbs to yield same source verb
    # here we already have tokens, we join??
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

  
# the use of split and join 
splitted_line = line.split()	# (line.split()) default delimiter = whitespace   # then remove stop words, etc
print(' '.join(splitted_line))


# Special model tokens:
PAD: padding (512 tokens for BERT)
UNK: unknown word 
CLS: start of a sentence 
SEP: sepearator at end of a sequence  (used in Q&A)
MASK: masking tokens (MLM=masked language model - used during training)
 
# Unicode normalization - NFD (cannonical decompostion) and NFC (cannonical decompostion followed by cannonical compostion) 
c_with_cedilla = '\u00C7'
c_plus_cedilla = '\u0043\u0327'
c_with_cedilla == c_plus_cedilla  # gives False
unicodedata.normalize('NFD', c_with_cedilla) == c_plus_cedilla # gives True
c_with_cedilla == unicodedata.normalize('NFC', c_plus_cedilla) # gives True
unicodedata.normalize('NFC', c_with_cedilla) == unicodedata.normalize('NFC', c_plus_cedilla) # gives True
# Normal form for Compatibility (K)
unicodedata.normalize('NFKD',"H")
fancy_h_with_cedilla = '\u210B\u0327'
h_with_cedilla = '\u1e28'
h_with_cedilla == fancy_h_with_cedilla  # gives False
unicodedata.normalize('NFKC',fancy_h_with_cedilla) == h_with_cedilla # gives True
https://ibm-learning.udemy.com/course/nlp-with-transformers/learn/lecture/25699794#overview
  
