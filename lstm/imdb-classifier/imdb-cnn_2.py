from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,MaxPooling1D
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import imdb
import numpy as np
from keras.preprocessing import text

wandb.init()
config = wandb.config

# set parameters:
config.vocab_size = 1000
#config.vocab_size = 500
config.maxlen = 1000
#config.batch_size = 32
config.batch_size = 64
config.embedding_dims = 50
config.filters = 250
#config.filters = 256
config.kernel_size = 3
config.hidden_dims = 250
#config.hidden_dims = 256
config.epochs = 10
config.dropout_rate=0.25

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()


tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print(len(X_train))
#print(X_train[0])

#exit()

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

#print(len(X_train))
#print(X_train[0])

#print(y_train[0])
#print(X_train.shape, y_train.shape)

model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen))
model.add(Dropout(config.dropout_rate))

#model.add(Conv1D(256, 
#                 3,
#                 input_shape=(25000,1000),
#                 padding='valid',
#                 activation='relu')) 

model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu')) 
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(config.dropout_rate))

#model.add(Conv1D(config.filters,
#                 config.kernel_size,
#                 padding='valid',
#                 activation='relu')) 
#model.add(MaxPooling1D(pool_size=2))
#model.add(Dropout(config.dropout_rate))
                    
model.add(Flatten())
model.add(Dense(config.hidden_dims, activation='relu'))
model.add(Dropout(config.dropout_rate))

#model.add(Dense(64, activation='relu'))
#model.add(Dropout(config.dropout_rate))


model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])
