

# To identify the similarity between tokens:
nlp = spacy.load(‘'en_core_web_lg'’)
tokens = nlp(sentence)
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

     
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

    
  
  
  
  
def print_similar_words(word, metric='cosine'):
   token = word_index.get(word)
   embedding = wights_embedding[token]
   distances = cdist(wights_embedding, [embedding], metric=metric).T[0]
   sorted_index = np.argsort(distances)
   sort_distances = distances[sorted_index]
   sorted_words = [reverse_word_index[token] for token in sorted_index if token !=0]
 
   def print_words(words, distances):
       for word, distance in zip(words, distances):
           print(distance, word)
 
   N = 10
   print(f'Distance from {word}')
   print_words(sorted_words[0:N], sort_distances[0:N])
   print("================")
   print_words(sorted_words[-N:], sort_distances[-N:])
 
print_similar_words('good', metric='cosine')



""" Gensim """
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format("word2vec_skipgram.txt", binary=False)
w2v.most_similar(positive=['solar'])
w2v.most_similar(positive=['kepler'])
