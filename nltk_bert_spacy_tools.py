
# ! git clone https://github.com/laxmimerit/twitter-disaster-prediction-dataset.git
url = 'https://raw.githubusercontent.com/laxmimerit/twitter-disaster-prediction-dataset/master/train.csv'
tweeter = pd.read_csv(url). # of size 7613 rows 
tweet = kgp.get_basic_features(tweeter)


""" ........... 1- NLTK ........... """
! pip install -U nltk		# (nltk = python modules + datasets)

# To download a particular dataset/models, use the nltk.download() function, e.g. if you are looking to download the punkt sentence tokenizer, use:
import nltk
nltk.download('punkt')  # Sometimes this is needed to use nltk.corpus.NAME(e.g. gutenberg).sents(‘book_name.txt’)

# If you're unsure of which data/model you need, you can start out with the basic list of data + models with:
nltk.download('popular')    # It will download a list of "popular" resources, these includes:
<collection id="popular" name="Popular packages">
      <item ref="cmudict" />
      <item ref="gazetteers" />
      <item ref="genesis" />
      <item ref="gutenberg" />
      <item ref="inaugural" />
      <item ref="movie_reviews" />
      <item ref="names" />
      <item ref="shakespeare" />
      <item ref="stopwords" />
      <item ref="treebank" />
      <item ref="twitter_samples" />
      <item ref="omw" />
      <item ref="wordnet" />
      <item ref="wordnet_ic" />
      <item ref="words" />
      <item ref="maxent_ne_chunker" />
      <item ref="punkt" />
      <item ref="snowball_data" />
      <item ref="averaged_perceptron_tagger" />
    </collection>

# stopwords    
from nltk.corpus import stopwords, twitter_samples
stopwords.words('english')[0:500:25]

# Part of Speech tagging 
from nltk import pos_tag
def _get_pos(text):
        pos=nltk.pos_tag(word_tokenize(text))
        pos=list(map(list,zip(*pos)))[1]
        return pos
      
# Name Entity Recognition NER via NLTK:
nltk.download('words')
nltk.download('maxent_ne_chunker')
from nltk import ne_chunk, pos_tag
chunked = ne_chunk(pos_tag(clean_tokens_list))
chunked.draw()

# The Brown Corpus was the first million-word electronic corpus of English, created in 1961 at Brown University
nltk.download('brown')
from nltk.corpus import brown
print(brown.sents())

# _________________________________________________________________________________________________________#

""" Gensim:                               https://radimrehurek.com/gensim/models/ldamodel.html
Gensim is a Python library that specializes in identifying semantic similarity between two documents through vector space modeling and topic modeling toolkit.
"""
# Ex) doc to bag of words bow 
from gensim.corpora import Dictionary     # Dictionary encapsulates the mapping between normalized words and their integer ids.
dct = Dictionary(["máma mele maso".split(), "ema má máma".split()])           # Dictionary(corpus)
dct.doc2bow(["this", "is", "máma"])      # dct.docebow(document (list of str) – Input document.)
# Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples.
dct.doc2bow(["this", "is", "máma"], return_missing=True)


# Ex) Train an Latent Dirichlet Allocation (LDA) model using a Gensim corpus
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
# Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=10)


# Ex) Word2Vec from gensim
# see word_embedding.py



# gensim lemmatizer ??
# wordnet lemmatizer 
# _________________________________________________________________________________________________________#
      
# Lemmatizer and Stemmer  
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('going', wordnet.VERB)

# 
S_stemmer = SnowballStemmer(language=”english”)       # better than PorterStemmer()
stemmer = PorterStemmer()
tokens = [stemmer.stem(t) for t in tokens]	where tokens = tokenized words                  Example: easilh -> easili 

# Tokenizers 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# Sentiment Analysis with NLTK
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer     # Give a sentiment intensity score to sentences.
sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(text)
ss.pop('compound')
key = ss.get


# Wordnet is an large, freely and publicly available lexical database for the English language aiming to establish structured semantic relationships between words
# It offers lemmatization capabilities as well and is one of the earliest and most commonly used lemmatizers
import nltk
nltk.download('wordnet')
nltk.download(‘vader_lexicon’)  # VADER sentiment analysis
nltk.download(“punkt”)	# download pre-trained punkt tokenizer for English

## 2. Wordnet Lemmatizer and 3. Wordnet Lemmatizer with appropriate POS tag
from nltk.stem.wordnet import WordNetLemmatizer
pass 






""" ....................................................... 2- TextBlob ....................................................... """ 
# # used for spelling corrections 
from textblob import TextBlob	# used for spelling corrections 

>> x = ‘sentence wiht topys’
>> x = TextBlob(x).correct() 



""" ...................................................... 3- Spacy ............................................................ """
# SpaCy offers the fastest syntactic parser available on the market today. Moreover, since the toolkit is written in Cython, it’s also really speedy and efficient
! python3 -m spacy download en_core_web_lg    #python3 -m spacy download en_core_web_sm
doc = nlp('sentence(s)')
# vec = doc.vector        # now, we need to get this vector in the form of numpy array 
tweet['vec'] = tweet['text'].apply(lambda x: nlp('x').vector)     # This is word2vec word embedding from Spacy 
X = tweet['vec'].to_numpy()
X = X.reshape(-1,1)
X = np.concatenate(np.concatenate(X,axis=0),axis=0).reshape(-1,300)           # X.shape = (7613,300)


doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')    # u for unicode
for token in doc:
  print(token.text, token.pos_, token.dep_,token.lemma_,token.tag_,token.dep_,token.shape_,token.is_alpha,token.is_stop) # Token.dep_ for dependencies, e.g., 'nsubj' for nominal subject
  spacy.explain('PROPN')
  spacy.explain('nsubj')


--------------------
# Tokenization with Spacy
[token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']
Another Example:
import spacy 
nlp = spacy.load('en_core_web_lg')        # nlp = spacy.load('en',disable=['parser', 'tagger','ner'])
nlp.max_length = 1198623
doc = nlp('x')    # x = 'cat dog 10km support@udemy.com'
print(type(doc), len(doc))  # spacy.tokens.doc.Doc
print(doc)  # cat \ndog\ \n10 \km \nsupport@udemy.com    == similar to tokenized words 

--------------------
# Lemmatization with Spacy (VIP: Stemming is not implemented in Spacy)
doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')    # u for unicode
for token in doc:
  print(token.text, '\t',token.lemma_,token.tag_,token.dep_,token.shape_,token.is_alpha,token.is_stop) # Token.dep_ for dependencies, e.g., 'nsubj' for nominal subject
  spacy.explain('PROPN')
  spacy.explain('nsubj')

--------------------
# stopwords with Spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
len(stopwords) 		# 26 

nlp = spacy.load('en_core_web_sm')
print(nlp.Defaults.stop_words)
print(len(nlp.Defaults.stop_words))       # 326
nlp.vocab['always'].is_stop   # returns true
nlp.Defaults.stop_words.add('asdf') # add 'asdf' from stopwords 
nlp.vocab['no'].is_stop = False     # remove 'no' from stopwords if set to True 
# or
nlp.Defaults.stop_words.remove('no')

--------------------
# vocabulary and phrase matching with spacy 
# 1- Rule based matcher explorer (under it - token based - rule based - phrase based )
https://spacy.io/usage/rule-based-matching/
https://explosion.ai/demos/matcher
# how to create this mathcer in python
import spacy 
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
# create patterns
pattern_1 = [{"LOWER": 'hello'}, {"LOWER": 'world'}]
pattern_2 = [{"LOWER": 'hello'}, {"IS_PUNCT": 'True'}, {"LOWER": 'world'}]
matcher.add("Hello World", None, pattern_1, pattern_2)
doc = nlp("'Hello World' are the first two printed words for most of the programmers")
find_matches = matcher(doc)
print(find_matches)           # returns the index of start/end of each match

from spacy.matcher import Matcher, PhraseMatcher
matcher = Matcher(nlp.vocab)
pattern1 = [{'LOWER': 'solarpower'}]            # pattern1 looks for a single token whose lowercase text reads 'solarpower'
pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]   # pattern2 looks for two adjacent tokens that read 'solar' and 'power' in that order
pattern3 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}] # pattern3 looks for three adjacent tokens, with a middle token that can be any punctuation.*
matcher.add('SolarPower', None, pattern1, pattern2, pattern3)	# matcher.remove('SolarPower')
find_matches = matcher(doc)
for match_id, start, end in find_matches:
      string_id = nlp.vocab.strings[match_id]  # get string representation
      span = doc[start:end]
      print(string_id, start, end, span.text)
To make the punctuation in the middle as optional, use ‘OP’:’*’ (I mean it does not matter wether punctuation exists or not)
pattern4 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]   # solar-power will be matched  
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LEMMA': 'power'}]

matcher = PhraseMatcher(nlp.vocab) 
phrase_list = ["Barak Obama", "Angela Markel", "Washinton", "D.C."]
phrase_patterns =  [nlp(text) for text in phrase_list]
matcher.add('Terminologyst', None, *phrase_patterns)
doc3 = nlp("German Chancellor Angela Markel and US president Barak Obama"
            "converse in the oval office inside the white house in Washinton D.C.")
find_matches = matcher(doc3)
print(find_matches)
for match_id, start, end in find_matches:
      string_id = nlp.vocab.strings[match_id]  # get string representation
      span = doc3[start:end]
      print(string_id, start, end, span.text)
      
--------------------
# Part of Speech Tagging PoS with spacy 
import spacy 
nlp = spacy.load('en_core_web_sm')
s1 = "Apple is looking at buying U.K. Startup for $1 billion"
doc = nlp(s1)
for token in doc:
      print(token.text, token.pos_, token.tag_, spacy.explain(token.tag_))
      # token.pos_ gives POS tag and token.tag_ gives fine-grained tag
for key, val in doc.count_by(spacy.attrs.POS).items():
      print(key, doc.vocab[key].text  val)
from spacy import displacy
displacy.render(doc=doc, style='dep', juypter=True, options={'distance': 100})
      
--------------------
# Named Entity Recognition NER with spacy 
s1 = "Apple is looking at buying U.K. Startup for $1 billion"
s2 = "San Francisco considers banning sidewalk delivery robots"
s3 = "facebook is hiring a new vice president in U.S."
nlp = spacy.load('en_core_web_sm')
doc1 = nlp(s1)
for ent in doc1.ents:
      print(ent.text, ent.label_, spacy.explain(ent.label_))   # Apple is ORG
doc3 = nlp(s3)
for ent in doc3.ents:
      print(ent.text, ent.label_, spacy.explain(ent.label_))   # facebook is not identified as ORG - doc3.ents = (U.S.,)
# define new objects
from spacy.tokens import Span
ORG = doc3.vocab.strings['ORG']
new_ent = Span(doc3, 0, 1, label=ORG)     # 0, 1 means first token which is for "facebook"
doc3.ents = list(doc3.ents) + [new_ent]
print(doc3.ents)  # (facebook, U.S.)
from spacy import displacy
displacy.render(docs=doc1, style='ent', juypter=True)
displacy.render(docs=doc1, style='ent', juypter=True, options={'ents': ['ORG']})    # to display organizations only

--------------------
# Sentence Segmentation with spacy 
from spacy.pipeline import SentenceSegmenter 
s1 = "This is a sentence. This is second sentence. This is last sentence."
s2 = "This is a sentence; This is second sentence; This is last sentence."
nlp = spacy.load('en_core_web_sm')
doc1 = nlp(s1)
# doc1.sents  # is a generator
for sent in doc1.sents:
      print(sent.text)
s3 = "This is a sentence. This is second U.K. sentence. This is last sentence."      
doc3 = nlp(s3)
for sent in doc3.sents:
      print(sent.text)  # spacy understands that U.K. is a word not end of a sentence

doc2 = nlp(s2)
for sent in doc2 .sents:      
      print(sent.text)        # as sentences are separated by ";", spacy returns all as one sentence

def set_custom_boundaries(doc):
      for token in doc[:-1]:
            if token.text == ';':
                  print(token.i)
                  doc[toekn.i+1].is_sent_start = True       # + 1 for the token after the ';'
      return doc
print(nlp.pipe_names)      #  ['tagger', 'parser', 'ner']
nlp.add_pipe(set_custom_boundaries, before='parser')
doc_2 = nlp(s2)
for sent in doc_2 .sents:      
      print(sent.text)

--------------------
# Pipeline with spacy 
nlp.pipeline
nlp.pipe_names		# ['tagger', 'parser', 'ner']
for sentence in doc4.sents:
    print(sentence)
doc4[6].is_sent_start		True
len(doc.vocab)
NER:  >> [(x.text,x.label_) for x in doc2.ents]

for ent in doc8.ents:		doc8.ents for name entities 
    print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))

doc9=nlp(‘sentence’)
doc9.noun_chunks		e.g., Autonomous cars	(nouns)

--------------------
# Displacy with spacy 
from spacy import displacy
doc = nlp(u'Apple is going to build a U.K. factory for $6 million.')
displacy.render(doc, style='dep', jupyter=True, options={'distance': 110}) 110 is the distance between tokens 
displacy.render(doc, style='ent', jupyter=True) # NER 
for entity in doc.ents:
  if entity.label_ == 'GPE':
    print(entity.text, entity.label_)
displacy.serve(doc, style=dep)



from spacy.tokens import span
for sent in doc.sents:		here doc.sents: is a generator, i.e., there is no doc.sents[0] in the memory. Instead, list(doc.sents)[0] (and the type is a span not actually a list).

    


      
Spacy features:
Tokenization - Lemmatization- PoS
Dependency Parsing 
SBD = sentence boundary detection 
NER and EL = entity linking 
Similarity and text classification
Rule-based matching (match a word to other words )
Training (pre trained by explosion - used to improve statistical models) and Serialization 
