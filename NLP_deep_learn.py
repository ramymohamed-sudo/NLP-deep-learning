
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

# to evaluate 
x = ['I am thrilled to see this']
x = get_clean(x)
x = toeknizer.texts_to_sequences(x)
vec = pad_sequences(x,max_len,padding='post')
np.argmax(model.predict(vec), axis=-1)




""" Sentiment Classification using BERT Model """
! pip install ktrain 
from ktrain import text
import ktrain
(x_train, y_train), (x_test,y_test), preproc = text.texts_from_df(train_df=tweet,text_column='text',label_columns='target',max_len=40,preprocess_mode='bert')
model = text.text_classifier(name='bert',train_data =(x_train, y_train), preproc = preproc)
learner = ktrain.get_learner(model=model, train_data = (x_train, y_train), val_data = (x_test,y_test), batch_size=64)
learner.fit_onecycle(lr=2e-5, epochs=2) # lr=2e-4
predictor = ktrain.get_predictor(learner.model,preproc)
data = ['I had car accident', 'I met him today by accident']
y_pred = predictor.predict(data[0], return_proba=True)
classes = predictor.get_classes()
classes.index(y_pred)   # 0 or 1 


""" Sentiment Classification using distil-BERT Model """
! git clone https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git
data_test = pd.read_excel('path',dtype=str)    
data_train = pd.read_excel('path',dtype=str)   
text.print_text_classifiers 
(train, , val, preproc) = text.texts_from_df(train_df = data_train,text_column='Reviews',label_columns='Sentiment',val_df = data_test,
                                             max_len=512,preprocess_mode='distilbert')    # 512 -> 400 to be faster 
model = text.text_classifier(name='distilbert',train_data =train, preproc = preproc)
learner = ktrain.get_learner(model=model, train_data = train, val_data = val, batch_size= 6)
learner.fit_onecycle(lr=2e-5, epochs=2) # lr=2e-4
predictor = ktrain.get_predictor(learner.model,preproc)
predictor.save('google drive/distilbert.h5')    # predictor = ktrain.load_predictor('')
y_pred = predictor.predict(data,return_proba=True)
classes = predictor.get_classes()











