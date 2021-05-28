

""" Part of Speech Tagging PoS"""

# Part of Speech (hereby referred to as POS) Tags are useful for building parse trees, which are used in building NERs 
# (most named entities are Nouns) and extracting relations between words. POS Tagging is also essential for building lemmatizers 
# which are used to reduce a word to its root form.

# Markov chain VIP: For speech recognition and POS tagging 

# POS tagging is critically important in search queries, speech recognition, and search


# because POS tags describe the characteristics structure of lexical terms in a sentence or text. You can use them to make assumptions about semantics. They're used for 
# identifying named entities too and a sentence
# Co-reference resolution, i,e., it refers to the Eiffel tower
# Speech recognition: to check if a sequence of words has a high probability or not



>>> w.lemmatize()		'octopus'
word("went").lemmatize("v")	 # Pass in WordNet part of speech (verb)

from nltk.tokenize import TweetTokenizer


# using nltk
from nltk.tokenize import word_tokenize
pos=nltk.pos_tag(word_tokenize(text))
#  tokenized_sentence = nltk.word_tokenize(sentence)		(== str.split())
# 
pos=list(map(list,zip(*pos)))[1]
# Name Entity Recognition NER via NLTK:
nltk.download('words')
nltk.download('maxent_ne_chunker')
tagged_words = nltk.pos_tag(words)
nemaedEnt = nltk.ne_chunk(tagged_words)
nemaedEnt.draw()




# using spacy
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')
for token in doc:
print(token.text, token.pos_, token.dep_)

from spacy.tokens import span 


# using TextBlob
# TextBlob aims to provide access to common text-processing operations through a familiar interface.		
# https://textblob.readthedocs.io/en/dev/quickstart.html
wiki = TextBlob("Python is a high-level, general-purpose programming language.")
 wiki.tags	#Part-of-speech tags can be accessed through the tags property
wiki.noun_phrases		#Noun Phrase Extraction
Wiki.sentiment: # The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.	Testimonial.sentiment.polarity
wiki.words 		# Tokenization
zen.sentences 	# returns sentence not words 
sentence.words[2].singularize()
sentence.words[-1].pluralize()
# Lemmantize:
# Lemmatization is the process of converting a word to its base form.
from textblob import Word
w = Word("octopi")
pos=nltk.pos_tag(word_tokenize(text))



Pattern Lemmatizer
Pattern allows part-of-speech tagging, sentiment analysis, vector space modeling, SVM, clustering, n-gram search, and WordNet. 
You can take advantage of a DOM parser, a web crawler, as well as some useful APIs like Twitter or Facebook. 
Still, the tool is essentially a web miner and might not be enough for completing other natural language processing tasks


TextBlob Lemmatizer
from textblob import Word
w = Word("octopi")
w.lemmatize()		'octopus'
word("went").lemmatize("v")	 # Pass in WordNet part of speech 

nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer  #  Wordnet Lemmatizer with appropriate POS tag

Gensim Lemmatize
Though we could not perform stemming with spaCy, we can perform lemmatization using spaCy. To do so, we need to use the lemma_ attribute on the spaCy document.

spacy
token.lemma gives number, token.lemma_ gives word such as “run” “be”
print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')
>> lemma = token.lemma_
>> if lemma = ‘-PRON-’ or lemma=’be’
>> lemma = token.text

Tokenization in Spacy:
text = 'Apple is looking for buying a UK startup for $1 billion dollar'
doc = nlp(text)
for token in doc:
    print(token.text)

  
Stemming
Porter stemmer
Snowball stemmer
Lancaster stemmer
Regex-based Stemmer


from nltk.stem import WordNetLemmatizer,PorterStemmer
	OR from nltk.stem.porter import PorterStemmer
	 from nltk.stem.snowball import SnowballStemmer
S_stemmer = SnowballStemmer(language=”english”)

>> stemmer = PorterStemmer()
>> tokens = [stemmer.stem(t) for t in tokens]	where tokens = words 




# Basic does for Pos
# 1- Using logisitic regression 
# 2- Using Recurrent NN
# 3- Using HMM



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




