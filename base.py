import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

train_path = os.path.join('jigsaw-toxic-comment-classification-challenge', 'train')
test_path = os.path.join('jigsaw-toxic-comment-classification-challenge', 'test')


train = pd.read_csv(os.path.join(train_path, 'train.csv'))
test = pd.read_csv(os.path.join(test_path,'test.csv'))
train.columns
# check for null characters

train.isnull().any()
test.isnull().any()

list_classes = ['toxic', 'severe_toxic', 'obscene',
                'threat','insult', 'identity_hate']

y = train[list_classes].values

list_sentences_train = train['comment_text']
list_sentences_test = test['comment_text']

max_features = 20000

tokenizer =Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# tokenizer.word_counts
# tokenizer.word_index

""" we use "padding"! We could make the shorter sentences as long as the others by filling 
the shortfall by zeros.But on the other hand, we also have to trim the longer
 ones to the same length(maxlen) as the short ones. In this case, we have set
  the max length to be 200."""

max_len = 200

X_t =pad_sequences(list_tokenized_train,maxlen=max_len)
X_te = pad_sequences(list_tokenized_test,maxlen=max_len)

inp = Input(shape=(max_len, ))

embed_size = 128
x = Embedding(max_features, embed_size)(inp)

x = LSTM(60, return_sequences=False,name='lstm_layer')(x)

x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
batch_size = 32
epochs = 2
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.summary()





# from keras import backend as K
#
# # with a Sequential model
# get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[2].output])
# layer_output = get_3rd_layer_output([X_t[:1]])[0]
# print(layer_output)

# print layer_output to see the actual data
for num1 in range(10,14):
    for num2 in range(10,14):
        print(num1, ",", num2)