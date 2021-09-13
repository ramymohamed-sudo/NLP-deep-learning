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



model_name = 'ProsusAI/finbert'
from transformers import BertForSequenceClassification 
from transformers import BertTokenizer
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

txt = ("Given the recent ....")
tokens = tokenizer.encode_plus(txt, max_length=512, truncation=True, padding=max_length, add_special_tokens=True, return_tensors='tf')  # 'pt' for Pytorch
[CLS] = 101, [SEP] = 102, [MASK] = 103, [UNK] = 100, [PAD] = 0, 
print(tokens)   # represented by a dict







