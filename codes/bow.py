
# some examples of bow are Word counting and TFIDF 


# https://www.freecodecamp.org/news/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04/
# bow can be seen as a special case of n-gram where n=1
# CBOW = extension of bigram 

# word2vec trains words against other words that neighbor them. To do so, it uses CBOW method, or the skip-gram method. 
Skip-gram uses auto-encoder input projection 
# Problem with BOW and TFIDF, they do not store the semantic information. 

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
BOW  = X.toarray()
vectorizer.get_feature_names()
bow = pd.DataFrame(BOW,columns=vectorizer.get_feature_names())
