
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
