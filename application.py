







""" ----------------------------------- Spacy  -----------------------------------  """
spacy
Spacy features:
Tokenization - Lemmatization- PoS
Dependency Parsing 
SBD = sentence boundary detection 
NER and EL = entity linking 
Similarity and text classification
Rule-based matching (match a word to other words )
Training (pre trained by explosion - used to improve statistical models) and Serialization 


""" ----------------------------------- NLTK  -----------------------------------  """
from nltk.sentiment.vader import SentimentIntensityAnalyzer   # Give a sentiment intensity score to sentences.



""" ----------------------------------- TEXTBLOB  -----------------------------------  """
# using TextBlob
# TextBlob aims to provide access to common text-processing operations through a familiar interface.		
# https://textblob.readthedocs.io/en/dev/quickstart.html

from textblob import TextBlob		# used for spelling corrections 
wiki = TextBlob("Python is a high-level, general-purpose programming language.")
wiki.sentiment: # The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.	Testimonial.sentiment.polarity




TextBlob for Translation, Tokenization, Sentiment Classifier:
tb = TextBlob(x)
tb.detect_language()		‘en’
tb.translate(to=’fr’)

>> TextBlob(x).words 		# x = sentence and this returns tokens 
>> from textblob.sentiments import NaiveBayesAnalyzer
>> x = ‘we …
>> tb = TextBlob(x,analyzer=’NaiveBayesAnalyzer’)
>> tb.sentiment		# pos 

pol = TextBlob(s).sentiment.polarity
TextBlob(df['art_title'][0]).sentiment.polarity
df['polarity'] = df['art_subhead'].map(lambda text: TextBlob(text).sentiment.polarity)
> df['binary_labels'] = df['label'].map({"ham":0,"spam":1})


TextBlob Sentiment estimator 
>> from textblob import TextBlob
Use the data from twitter 
>> raw_twitts = json.loads(data)
>> x = str(raw_twitts[‘text’]).lower()
>> blob = TextBlob(x)
>> blob.sentiment.polarity			# print(blob.sentiment) 
7. Pattern 


