
# 1- 
# Wikipedia size is 12 GB. You can download chunks of size 100x MB, still need to do parsing as it is XML format.



# 2- Alternative is the brown Corpus - comes with nltk 
# nltk.download('brown')
from nltk.corpus import brown


def get_sentences():
  # returns 57340 of the Brown corpus (around 57340 sentences) - much smaller than Wikipedia but OK for the course 
  # each sentence is represented as a list of individual string tokens
  return brown.sents()      # list of lists, each inner list is a sentence = list of words already tokenized 


