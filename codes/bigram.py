
#  Build a language model using Bi-gram
# prob of current word | last word: prob(cat|the)

# https://ibm-learning.udemy.com/course/natural-language-processing-with-deep-learning-in-python/learn/lecture/9489224#overview
# 1- see the Markov model
# 2- see the nueral version of the same Markov model: close connection between Markov model and Logistic regression 
# relation between bi-gram and word2vec

# language model= a model of the probability of a sequence of words (e.g., a sentence)
# Ex: P('the quicl brown fox jumps over the lazy dog')
# bigram is a specific language model
# P(B|A) = prob(A,B)/Prob(B) = count(A->B)/count(A)

# how to calculate P(sentence) based on the bi-gram model; make some assumptions 
# apply Bayes rule P(C|B|A) = P(C|A,B)*P(B|A) = P(C|A,B)*P(B|A)*p(A)
# is called chain rule of probabilities 
# P(B|A) is a bi-gram - the rest are not bigrams, but can be calculated using maximum likelehhod 
# P(A) = count(A)/corpus length
# Trigram P(C|A,B) = count(A->B->C)/count(A->B)

# if a sentence never showed up in a sentence, its maximum likelehood prob is 0. It is inaccurate to be 0 as in English, 
# the sentence P('the quicl brown fox jumps over the lazy dog') makes complete sense 
# Soln: smoothing +1 in num and +V in den 

# Markov assumption P(E|A,B,C,D) = P(E|D). Hence, P(A,B,C,D,E) = P(E|D)P(D|C)P(C|B)P(B|A)P(A)

# the brown corpus is used in this code 
from nltk.corpus import brown
def get_sentences():
  # returns 57340 of the Brown corpus
  # each sentence is represented as a list of individual string tokens
  return brown.sents()

# assign a unique integer for every word, i.e., this fn maps from word representation to word representation
def get_sentences_with_word2idx():
  sentences = get_sentences()
  indexed_sentences = []
  i = 2
  word2idx = {'START': 0, 'END': 1}
  for sentence in sentences:
    indexed_sentence = []
    for token in sentence:
      token = token.lower()
      if token not in word2idx:
        word2idx[token] = i
        i += 1
      indexed_sentence.append(word2idx[token])
    indexed_sentences.append(indexed_sentence)
  print("Vocab size:", i)
  return indexed_sentences, word2idx
