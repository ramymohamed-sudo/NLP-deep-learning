
from nltk.corpus import brown

def get_sentences():
  # returns 57340 of the Brown corpus
  # each sentence is represented as a list of individual string tokens
  return brown.sents()
