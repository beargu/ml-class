import amazon
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, GRU
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.preprocessing import text
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config
config.vocab_size = 1000

#config.maxlen = 1000
config.maxlen = 500
config.batch_size = 32
config.embedding_dims = 50
#config.filters = 250
config.filters = 128
config.kernel_size = 3
config.hidden_dims = 250
config.epochs = 10
config.dropout_rate=0.25


(train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels) = amazon.load_amazon()
print(train_summary[1])
print(train_review_text[1])
print(len(train_review_text))
#exit()

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(train_review_text)
X_train = tokenizer.texts_to_matrix(train_review_text)
X_test = tokenizer.texts_to_matrix(test_review_text)

y_train = train_labels
y_test = test_labels

print(y_train[0:10])
print(y_test[0:10])
#print(X_train[0])
# Build the model : original
#model = Sequential()
#model.add(Dense(1, activation='softmax', input_shape=(config.vocab_size,)))

#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
#model.fit(X_train, train_labels, epochs=10, validation_data=(X_test, test_labels),
#    callbacks=[WandbCallback()])





####
# try other models


X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)


model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen))
model.add(Dropout(config.dropout_rate))



model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu')) 
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(config.dropout_rate))

model.add(GRU(config.hidden_dims, activation="sigmoid", dropout=0.2))

                    
#model.add(Flatten())
#model.add(Dense(config.hidden_dims, activation='relu'))
#model.add(Dropout(config.dropout_rate))

model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])

