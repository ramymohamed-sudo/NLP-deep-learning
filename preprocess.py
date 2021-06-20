
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
  x = str(x).lower().replace('\\','').replace('_',' ').replace('.',' ')
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
    # stopw = stopwords.word('english')
    # tokens = [token for token in tokens if token not in stopw]
    # remove words less than 3 words
    # tokens = [word for word in tokens if len(word) >=3]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]
    # here we already have tokens, we join??
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
