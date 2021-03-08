
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 

# ! git clone https://github.com/laxmimerit/twitter-disaster-prediction-dataset.git
url = 'https://raw.githubusercontent.com/laxmimerit/twitter-disaster-prediction-dataset/master/train.csv'
tweeter = pd.read_csv(url)    # 7613 rows 
tweet = kgp.get_basic_features(tweeter)

""" Tokenizer from tensorflow.keras.preprocessing.text """
MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
# If you do not want to filter special characters Tokenizer(num_words = MAX_VOCAB_SIZE,filters=’’)		e.g., want to keep ‘<sos>’ and ‘<eos>’ 
# tokenizer.fit_on_texts(sentences) 		
toeknizer.fit_on_texts(tweet['text'])
print(toeknizer.word_index)
vocab_size = len(toeknizer.word_index) + 1
encoded_text = toeknizer.texts_to_sequences(tweet['text'])  # they are numbers    # sequences_test = toeknizer.texts_to_sequences(df_test)  
print(encoded_text)
toeknizer.word_counts 
toeknizer.word_index.items()

max_len = 40 # words
X = pad_sequences(encoded_text, maxlen = MAX_SEQUENCE_LENGTH,padding='post',value=0)   # 7613 rows and 40 columns after padding 
y = tweet['target']
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42, stratify=y) 


""" Convolutional NN """
model = sequential()
model.add( Embedding(vocab_size=vocab_size,vec_size=100, input_length=max_len, d_feature=d_model) )
model.add( Conv1D(32,2, activation='relu') )
model.add( Maxpooling1D(2) )
model.add( Dense(32, activation='relu') )
model.add( Dropout(0.5))
model.add( Dense(16, activation='relu') )
model.add( GlobalMaxpooling1D() )
model.add( Dense(1, activation='sigmoid') )
model.summary
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))






