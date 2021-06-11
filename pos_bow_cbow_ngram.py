
n-gram and bow as a special case of n-gram when n =1
bow vs cbow 
n-gram and language model
n-gram codes and its applications 


""" ----------------------------------- Stopwords and word.lower()  -----------------------------------  """

""" ----------------------------------- Padding  -----------------------------------  """
# nltk
>> from nltk.corpus import twitter_samples, stopwords
>> from nltk.util import ngrams, pad_sequence
	s = list(ngrams(['This' ,'is','an','NLP','course','at', 'OReilly'],2))

>>pad_sequence(text[0],
... pad_left=True,
... left_pad_symbol="<s>",
... pad_right=True,
... right_pad_symbol="</s>",
... n=2)

>> punct = set(string.punctuation)



# Include special tokens 
# started with pad, end of line and unk tokens
Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 


from tensorflow.keras.preprocessing.sequence import pad_sequences
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH,padding='post',value=0)



""" ----------------------------------- Tokenization  -----------------------------------  """

# nltk 
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
word_tokenize(text)

# Tokenization in Spacy:
text = 'Apple is looking for buying a UK startup for $1 billion dollar'
doc = nlp(text)
for token in doc:
    print(token.text)

# Textblob
wiki = TextBlob("Python is a high-level, general-purpose programming language.")
wiki.words 		# Tokenization
wiki.sentences 	# returns sentence not words 


# keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds #(for tokenizers)
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences) 		

> tokenizer.word_index
> MAX_SEQUENCE_LENGTH =5 
> data = pad_sequences(sequences,
maxlen = MAX_SEQUENCE_LENGTH,padding='post',value=0)


""" -----------------------------------  Part of Speech Tagging PoS  ----------------------------------- """
# Part of Speech (hereby referred to as POS) Tags are useful for building parse trees, which are used in building NERs 
# (most named entities are Nouns) and extracting relations between words. POS Tagging is also essential for building lemmatizers 
# which are used to reduce a word to its root form.

# Markov chain VIP: For speech recognition and POS tagging 

# POS tagging is critically important in search queries, speech recognition, and search

# because POS tags describe the characteristics structure of lexical terms in a sentence or text. You can use them to make assumptions about semantics. They're used for 
# identifying named entities too and a sentence
# Co-reference resolution, i,e., it refers to the Eiffel tower
# Speech recognition: to check if a sequence of words has a high probability or not

from nltk.tokenize import word_tokenize
pos=nltk.pos_tag(word_tokenize(text))
#  tokenized_sentence = nltk.word_tokenize(sentence)		(== str.split())
# 
pos=list(map(list,zip(*pos)))[1]


# using spacy
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')
for token in doc:
print(token.pos_, token.dep_)		# token.text is the token 

# creating a span object of spacy
# from spacy.tokens import span 


# using TextBlob
# TextBlob aims to provide access to common text-processing operations through a familiar interface.		
# https://textblob.readthedocs.io/en/dev/quickstart.html
wiki = TextBlob("Python is a high-level, general-purpose programming language.")
wiki.tags	#Part-of-speech tags can be accessed through the tags property
wiki.noun_phrases		# Noun Phrase Extraction
Wiki.sentiment: # The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.	Testimonial.sentiment.polarity
sentence.words[2].singularize()
sentence.words[-1].pluralize()


# Basic does for Pos
# 1- Using logisitic regression 
# 2- Using Recurrent NN
# 3- Using HMM

""" -----------------------------------  Chunking  ----------------------------------- """
nltk.download('maxent_ne_chunker')

nemaedEnt = nltk.ne_chunk(tagged_words)

doc9=nlp(‘sentence’)
doc9.noun_chunks		e.g., Autonomous cars	(nouns)



""" ----------------------------------- Lemmatization  -----------------------------------  """
# Pattern Lemmatizer
# Pattern allows part-of-speech tagging, sentiment analysis, vector space modeling, SVM, clustering, n-gram search, and WordNet. 
# You can take advantage of a DOM parser, a web crawler, as well as some useful APIs like Twitter or Facebook. 
# Still, the tool is essentially a web miner and might not be enough for completing other natural language processing tasks

# nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer  #  Wordnet Lemmatizer with appropriate POS tag

# TextBlob Lemmatizer
from textblob import Word
w = Word("octopi")
w.lemmatize()		# 'octopus'
word("went").lemmatize("v")	 # Pass in WordNet part of speech 


# Gensim Lemmatize
gensim.utils.lemmatize(content, allowed_tags=<_sre.SRE_Pattern object>, light=False, stopwords=frozenset([]), min_length=2, max_length=15)
lemmatize('Hello World! How is it going?! Nonexistentword, 21')   # ['world/NN', 'be/VB', 'go/VB', 'nonexistentword/NN']

# spacy
# Though we could not perform stemming with spaCy, we can perform lemmatization using spaCy. To do so, we need to use the lemma_ attribute on the spaCy document.
token.lemma gives number, token.lemma_ gives word such as “run” “be”
print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')
lemma = token.lemma_
if lemma = ‘-PRON-’ or lemma=’be’
lemma = token.text

# CLiPS Pattern
# Stanford CoreNLP
# TreeTagger


""" -------------------------------------- Stemming --------------------------------------  """
# Porter stemmer
# Snowball stemmer
# Lancaster stemmer
# Regex-based Stemmer

# nltk 
from nltk.stem import PorterStemmer	# OR from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokens = [stemmer.stem(t) for t in tokens]	# where tokens = words 

from nltk.stem.snowball import SnowballStemmer
S_stemmer = SnowballStemmer(language=”english”)


""" ----------------------------------- Name Entity Recoginition  -----------------------------------  """

# NLTK:
nltk.download('words')
nltk.download('maxent_ne_chunker')
tagged_words = nltk.pos_tag(words)
nemaedEnt = nltk.ne_chunk(tagged_words)
nemaedEnt.draw()


""" ----------------------------------- Dependency  -----------------------------------  """



""" ----------------------------------- Bag of words  -----------------------------------  """
# Bag of Words just creates a set of vectors containing the count of word occurrences in the document (reviews), 
# while the TF-IDF model contains information on the more important words and the less important ones as well.

# text classification using BOW features based on word vectors from word2vec and glove 

# It is worth noting that the word embeddings of word2vec were created by a continuous skip-gram model, and not a continuous bag of words model.
# word2vec trains words against other words that neighbor them. To do so, it uses CBOW method, or the skip-gram method. 
# word2vec trains words against other words that neighbor them. To do so, it uses CBOW method, or the skip-gram method. 
# cbow = extension of 

# CBOW = extension of bigram 
# BOW and TF (Bag of Words and Term Frequency) 

# Problem with BOW and TFIDF, they do not store the semantic information. 


# CountVectorizer implements both tokenization and occurrence counting in a single class:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray(),type(X.toarray()),X.shape)


Bag of Words is a commonly used model that depends on word frequencies or occurrences to train a classifier. 
This model creates an occurrence matrix for documents or sentences irrespective of its grammatical structure or word order. 


>> CountVectorizer implements both tokenization and occurrence counting in a single class:
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
BOW  = X.toarray()
vectorizer.get_feature_names()
bow = pd.DataFrame(BOW,columns=vectorizer.get_feature_names())

tfidf = TfidfVectorizer()
norm=l1 (default =l2)
ngram_range=(1,2)		(default=1,1)
analyzer=’word’		tokenization done word by word (‘char’)
max_features = 5000 (limit the dictionary to 5000)


feature_matrix = tfidf.fit_transform(body_all_articles)
feature_matrix.toarray()


""" -------------------------------------- Continuous Bag of Words  -----------------------------------  """



""" ----------------------------------- Skip Gram and N-Gram extraction  -----------------------------------  """



Basic Dependency Grammar




Dependency Parsing and Constituency Parsing
Dependency Parsing, also known as Syntactic parsing in NLP is a process of assigning syntactic structure to a sentence and identifying its dependency parses. This process is crucial to understand the correlations between the “head” words in the syntactic structure.
The process of dependency parsing can be a little complex considering how any sentence can have more than one dependency parses. Multiple parse trees are known as ambiguities. Dependency parsing needs to resolve these ambiguities in order to effectively assign a syntactic structure to a sentence.

Dependency parsing can be used in the semantic analysis of a sentence apart from the syntactic structuring.

# Dependency Visualization 
>> from spacy import displacy
 >> displacy.render(doc3, style='ent')
>> displacy.render(doc, style='dep',options={'distance':100,'compact':True})








