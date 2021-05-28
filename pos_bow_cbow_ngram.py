

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


>>> w.lemmatize()		'octopus'
word("went").lemmatize("v")	 # Pass in WordNet part of speech (verb)


# using nltk
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


# Basic does for Pos
# 1- Using logisitic regression 
# 2- Using Recurrent NN
# 3- Using HMM



""" Bag of words """
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




