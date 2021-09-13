!pip install flair 
import flair 
model = flair.models.TextClassifier.load('en-sentiment')
text = 'I like you!'
sentence = flair.data.sentence(text)
sentence.to_tokenized_string()
model.predict(sentence)
sentence    # label is given +ve 
sentence.get_labels()[0]    # type is flair data - has attributes such as score and value
sentence.get_labels()[0].score
help(sentence.get_labels()[0].value)

# hugging face transformer library 
# most advanced and accesible library for transofrmer 







